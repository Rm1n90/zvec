// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <gtest/gtest.h>
#include "db/index/column/fts_column/fts_posting_codec.h"


using namespace zvec;


TEST(FtsPostingCodecTest, EmptyRoundTrip) {
  std::string encoded;
  ASSERT_TRUE(FtsPostingCodec::Encode({}, &encoded).ok());

  std::vector<FtsPosting> decoded;
  ASSERT_TRUE(FtsPostingCodec::Decode(encoded.data(), encoded.size(), &decoded)
                  .ok());
  EXPECT_TRUE(decoded.empty());
}


TEST(FtsPostingCodecTest, SinglePostingRoundTrip) {
  std::vector<FtsPosting> in{{42, 3}};
  std::string encoded;
  ASSERT_TRUE(FtsPostingCodec::Encode(in, &encoded).ok());

  std::vector<FtsPosting> out;
  ASSERT_TRUE(
      FtsPostingCodec::Decode(encoded.data(), encoded.size(), &out).ok());
  EXPECT_EQ(out, in);
}


TEST(FtsPostingCodecTest, MultiplePostingRoundTrip) {
  std::vector<FtsPosting> in{{1, 1}, {7, 2}, {200, 5}, {20000, 1}};
  std::string encoded;
  ASSERT_TRUE(FtsPostingCodec::Encode(in, &encoded).ok());

  std::vector<FtsPosting> out;
  ASSERT_TRUE(
      FtsPostingCodec::Decode(encoded.data(), encoded.size(), &out).ok());
  EXPECT_EQ(out, in);
}


TEST(FtsPostingCodecTest, EncodeSingleRoundTrip) {
  auto encoded = FtsPostingCodec::EncodeSingle(12345, 7);
  std::vector<FtsPosting> out;
  ASSERT_TRUE(
      FtsPostingCodec::Decode(encoded.data(), encoded.size(), &out).ok());
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].doc_id, 12345u);
  EXPECT_EQ(out[0].tf, 7u);
}


TEST(FtsPostingCodecTest, EncodeRejectsUnsortedInput) {
  std::vector<FtsPosting> in{{5, 1}, {3, 1}};
  std::string encoded;
  auto s = FtsPostingCodec::Encode(in, &encoded);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
}


TEST(FtsPostingCodecTest, EncodeRejectsDuplicates) {
  std::vector<FtsPosting> in{{5, 1}, {5, 2}};
  std::string encoded;
  auto s = FtsPostingCodec::Encode(in, &encoded);
  EXPECT_FALSE(s.ok());
}


TEST(FtsPostingCodecTest, DecodeRejectsTruncatedCount) {
  // A single 0xFF byte encodes a varint that continues past the end.
  const char buf[] = {'\xFF'};
  std::vector<FtsPosting> out;
  auto s = FtsPostingCodec::Decode(buf, sizeof(buf), &out);
  EXPECT_FALSE(s.ok());
}


TEST(FtsPostingCodecTest, DecodeRejectsTruncatedPosting) {
  // count = 2, first doc_id = 1, first tf = 1, second doc_id delta is missing.
  std::string buf;
  buf.push_back('\x02');  // count
  buf.push_back('\x01');  // doc_id_0 = 1
  buf.push_back('\x01');  // tf_0 = 1
  // truncated here
  std::vector<FtsPosting> out;
  auto s = FtsPostingCodec::Decode(buf.data(), buf.size(), &out);
  EXPECT_FALSE(s.ok());
}


TEST(FtsPostingCodecTest, DecodeRejectsTrailingBytes) {
  auto encoded = FtsPostingCodec::EncodeSingle(1, 1);
  encoded.push_back('\x00');  // trailing garbage
  std::vector<FtsPosting> out;
  auto s = FtsPostingCodec::Decode(encoded.data(), encoded.size(), &out);
  EXPECT_FALSE(s.ok());
}


TEST(FtsPostingCodecTest, DecodeRejectsDuplicateDocId) {
  // count = 2, doc_id_0 = 5, tf_0 = 1, doc_id_1 delta = 0 (duplicate), tf_1 = 1
  std::string buf;
  buf.push_back('\x02');
  buf.push_back('\x05');
  buf.push_back('\x01');
  buf.push_back('\x00');
  buf.push_back('\x01');
  std::vector<FtsPosting> out;
  auto s = FtsPostingCodec::Decode(buf.data(), buf.size(), &out);
  EXPECT_FALSE(s.ok());
}


TEST(FtsPostingCodecTest, MergeEncodedNoOverlap) {
  std::string a;
  std::string b;
  ASSERT_TRUE(FtsPostingCodec::Encode({{1, 1}, {3, 1}}, &a).ok());
  ASSERT_TRUE(FtsPostingCodec::Encode({{2, 1}, {4, 1}}, &b).ok());

  std::string merged;
  ASSERT_TRUE(FtsPostingCodec::MergeEncoded(a.data(), a.size(), b.data(),
                                            b.size(), &merged)
                  .ok());
  std::vector<FtsPosting> out;
  ASSERT_TRUE(
      FtsPostingCodec::Decode(merged.data(), merged.size(), &out).ok());
  std::vector<FtsPosting> expected{{1, 1}, {2, 1}, {3, 1}, {4, 1}};
  EXPECT_EQ(out, expected);
}


TEST(FtsPostingCodecTest, MergeEncodedOverlapSumsTf) {
  std::string a;
  std::string b;
  ASSERT_TRUE(FtsPostingCodec::Encode({{1, 1}, {2, 3}}, &a).ok());
  ASSERT_TRUE(FtsPostingCodec::Encode({{2, 4}, {5, 7}}, &b).ok());

  std::string merged;
  ASSERT_TRUE(FtsPostingCodec::MergeEncoded(a.data(), a.size(), b.data(),
                                            b.size(), &merged)
                  .ok());
  std::vector<FtsPosting> out;
  ASSERT_TRUE(
      FtsPostingCodec::Decode(merged.data(), merged.size(), &out).ok());
  std::vector<FtsPosting> expected{{1, 1}, {2, 7}, {5, 7}};
  EXPECT_EQ(out, expected);
}


TEST(FtsPostingCodecTest, MergeSingleIntoAppendsAtEnd) {
  std::vector<FtsPosting> base{{1, 1}, {2, 1}};
  ASSERT_TRUE(FtsPostingCodec::MergeSingleInto(&base, 5, 3).ok());
  std::vector<FtsPosting> expected{{1, 1}, {2, 1}, {5, 3}};
  EXPECT_EQ(base, expected);
}


TEST(FtsPostingCodecTest, MergeSingleIntoInsertsInMiddle) {
  std::vector<FtsPosting> base{{1, 1}, {5, 1}};
  ASSERT_TRUE(FtsPostingCodec::MergeSingleInto(&base, 3, 2).ok());
  std::vector<FtsPosting> expected{{1, 1}, {3, 2}, {5, 1}};
  EXPECT_EQ(base, expected);
}


TEST(FtsPostingCodecTest, MergeSingleIntoMergesExisting) {
  std::vector<FtsPosting> base{{1, 1}, {5, 2}};
  ASSERT_TRUE(FtsPostingCodec::MergeSingleInto(&base, 5, 3).ok());
  std::vector<FtsPosting> expected{{1, 1}, {5, 5}};
  EXPECT_EQ(base, expected);
}


TEST(FtsPostingCodecTest, MergeSingleIntoEmptyBase) {
  std::vector<FtsPosting> base;
  ASSERT_TRUE(FtsPostingCodec::MergeSingleInto(&base, 7, 2).ok());
  std::vector<FtsPosting> expected{{7, 2}};
  EXPECT_EQ(base, expected);
}


TEST(FtsPostingMergerTest, FullMergeNoExistingSingleOperandCopiesThrough) {
  auto op = FtsPostingCodec::EncodeSingle(42, 5);
  std::vector<rocksdb::Slice> operands{rocksdb::Slice(op)};
  rocksdb::Slice key{};

  rocksdb::MergeOperator::MergeOperationInput in(key, nullptr, operands,
                                                 nullptr);
  std::string new_value;
  rocksdb::Slice existing_operand;
  rocksdb::MergeOperator::MergeOperationOutput out(new_value, existing_operand);

  FtsPostingMerger merger;
  ASSERT_TRUE(merger.FullMergeV2(in, &out));
  EXPECT_EQ(new_value, op);
}


TEST(FtsPostingMergerTest, FullMergeWithExistingAndOperand) {
  std::string existing;
  ASSERT_TRUE(FtsPostingCodec::Encode({{1, 1}, {3, 2}}, &existing).ok());
  auto op = FtsPostingCodec::EncodeSingle(2, 4);
  std::vector<rocksdb::Slice> operands{rocksdb::Slice(op)};
  rocksdb::Slice existing_slice(existing);
  rocksdb::Slice key{};

  rocksdb::MergeOperator::MergeOperationInput in(key, &existing_slice, operands,
                                                 nullptr);
  std::string new_value;
  rocksdb::Slice existing_operand;
  rocksdb::MergeOperator::MergeOperationOutput out(new_value, existing_operand);

  FtsPostingMerger merger;
  ASSERT_TRUE(merger.FullMergeV2(in, &out));

  std::vector<FtsPosting> decoded;
  ASSERT_TRUE(
      FtsPostingCodec::Decode(new_value.data(), new_value.size(), &decoded)
          .ok());
  std::vector<FtsPosting> expected{{1, 1}, {2, 4}, {3, 2}};
  EXPECT_EQ(decoded, expected);
}


TEST(FtsPostingMergerTest, FullMergeMultipleOperandsSumTf) {
  auto op1 = FtsPostingCodec::EncodeSingle(10, 1);
  auto op2 = FtsPostingCodec::EncodeSingle(10, 2);
  auto op3 = FtsPostingCodec::EncodeSingle(11, 5);
  std::vector<rocksdb::Slice> operands{rocksdb::Slice(op1), rocksdb::Slice(op2),
                                       rocksdb::Slice(op3)};
  rocksdb::Slice key{};

  rocksdb::MergeOperator::MergeOperationInput in(key, nullptr, operands,
                                                 nullptr);
  std::string new_value;
  rocksdb::Slice existing_operand;
  rocksdb::MergeOperator::MergeOperationOutput out(new_value, existing_operand);

  FtsPostingMerger merger;
  ASSERT_TRUE(merger.FullMergeV2(in, &out));

  std::vector<FtsPosting> decoded;
  ASSERT_TRUE(
      FtsPostingCodec::Decode(new_value.data(), new_value.size(), &decoded)
          .ok());
  std::vector<FtsPosting> expected{{10, 3}, {11, 5}};
  EXPECT_EQ(decoded, expected);
}


TEST(FtsPostingMergerTest, PartialMergeTwoOperands) {
  std::string a;
  std::string b;
  ASSERT_TRUE(FtsPostingCodec::Encode({{1, 1}, {3, 1}}, &a).ok());
  ASSERT_TRUE(FtsPostingCodec::Encode({{3, 2}, {5, 1}}, &b).ok());

  FtsPostingMerger merger;
  std::string new_value;
  ASSERT_TRUE(
      merger.PartialMerge(rocksdb::Slice{}, rocksdb::Slice(a), rocksdb::Slice(b),
                          &new_value, nullptr));

  std::vector<FtsPosting> decoded;
  ASSERT_TRUE(FtsPostingCodec::Decode(new_value.data(), new_value.size(),
                                      &decoded)
                  .ok());
  std::vector<FtsPosting> expected{{1, 1}, {3, 3}, {5, 1}};
  EXPECT_EQ(decoded, expected);
}


TEST(FtsPostingMergerTest, NameIsStable) {
  FtsPostingMerger merger;
  EXPECT_STREQ(merger.Name(), "FtsPostingMerger");
}
