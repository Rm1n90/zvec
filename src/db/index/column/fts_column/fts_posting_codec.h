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


#pragma once


#include <cstdint>
#include <string>
#include <vector>
#include <rocksdb/merge_operator.h>
#include <rocksdb/slice.h>
#include <zvec/db/status.h>


namespace zvec {


/**
 * @brief A single posting: (doc_id, term frequency in that doc).
 */
struct FtsPosting {
  uint32_t doc_id{0};
  uint32_t tf{0};

  bool operator==(const FtsPosting &other) const {
    return doc_id == other.doc_id && tf == other.tf;
  }
};


/**
 * @brief Encode / decode / merge BM25-style posting lists.
 *
 * Wire format (all integers are LEB128 varints):
 *
 *     [N]
 *     [doc_id_0]                       // delta from 0
 *     [tf_0]
 *     [doc_id_1 - doc_id_0]            // delta
 *     [tf_1]
 *     ...
 *     [doc_id_{N-1} - doc_id_{N-2}]
 *     [tf_{N-1}]
 *
 * Postings are stored in strictly ascending order of doc_id with no
 * duplicates. Delta encoding keeps the format compact when doc ids are
 * clustered — a common case when many docs are inserted in a batch.
 */
class FtsPostingCodec {
 public:
  /// Serialize @p postings into @p out (cleared first).
  /// Requires @p postings to be sorted by doc_id ascending and have no
  /// duplicates. Returns InvalidArgument if that contract is violated.
  static Status Encode(const std::vector<FtsPosting> &postings,
                       std::string *out);

  /// Serialize a single posting — a fast-path used by the indexer's insert
  /// path to avoid allocating a vector for every Merge() operand.
  static std::string EncodeSingle(uint32_t doc_id, uint32_t tf);

  /// Deserialize a posting list from @p data / @p size into @p out.
  /// @p out is cleared first.
  static Status Decode(const char *data, std::size_t size,
                       std::vector<FtsPosting> *out);

  /// Merge two encoded posting lists into @p out.
  /// If the same doc_id appears in both, the resulting tf is the sum.
  /// This matches the write-path semantics where each insert is additive.
  static Status MergeEncoded(const char *lhs, std::size_t lhs_size,
                             const char *rhs, std::size_t rhs_size,
                             std::string *out);

  /// Merge an encoded list with the in-memory operand produced by
  /// EncodeSingle(). Slightly faster than MergeEncoded() when we know one
  /// side has exactly one posting.
  static Status MergeSingleInto(std::vector<FtsPosting> *base,
                                uint32_t doc_id, uint32_t tf);
};


/**
 * @brief RocksDB Merge operator for full-text posting lists.
 *
 * Analogous to @ref InvertedRocksdbValueMerger, but for posting lists that
 * carry term-frequencies — the existing roaring-bitmap merger is a pure set
 * union and loses the tf information BM25 needs.
 */
class FtsPostingMerger : public rocksdb::MergeOperator {
 public:
  bool FullMergeV2(const MergeOperationInput &merge_in,
                   MergeOperationOutput *merge_out) const override;
  bool PartialMerge(const rocksdb::Slice &key,
                    const rocksdb::Slice &left_operand,
                    const rocksdb::Slice &right_operand, std::string *new_value,
                    rocksdb::Logger *logger) const override;
  const char *Name() const override;
};


}  // namespace zvec
