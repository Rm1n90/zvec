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


#include "fts_posting_codec.h"

#include <algorithm>
#include <zvec/ailego/logger/logger.h>


namespace zvec {


namespace {

// Write a 32-bit unsigned integer as LEB128 varint.
inline void write_varint(uint32_t value, std::string *out) {
  while (value >= 0x80) {
    out->push_back(static_cast<char>(static_cast<uint8_t>(value) | 0x80));
    value >>= 7;
  }
  out->push_back(static_cast<char>(value));
}

// Read a 32-bit unsigned integer from LEB128 varint.
// On success advances @p pos and stores the value in @p out.
// Returns false if the input is truncated or the value overflows uint32.
inline bool read_varint(const char *data, std::size_t size, std::size_t *pos,
                        uint32_t *out) {
  uint64_t result = 0;
  int shift = 0;
  while (*pos < size) {
    const uint8_t byte = static_cast<uint8_t>(data[*pos]);
    ++(*pos);
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      if (result > 0xFFFFFFFFULL) {
        return false;
      }
      *out = static_cast<uint32_t>(result);
      return true;
    }
    shift += 7;
    if (shift >= 35) {
      return false;
    }
  }
  return false;
}

}  // namespace


Status FtsPostingCodec::Encode(const std::vector<FtsPosting> &postings,
                               std::string *out) {
  if (out == nullptr) {
    return Status::InvalidArgument("out is null");
  }
  out->clear();

  // Validate ordering.
  for (std::size_t i = 1; i < postings.size(); ++i) {
    if (postings[i].doc_id <= postings[i - 1].doc_id) {
      return Status::InvalidArgument(
          "FtsPostingCodec::Encode: postings must be sorted by doc_id "
          "ascending with no duplicates");
    }
  }

  out->reserve(1 + postings.size() * 4);
  write_varint(static_cast<uint32_t>(postings.size()), out);
  uint32_t prev = 0;
  for (const FtsPosting &p : postings) {
    write_varint(p.doc_id - prev, out);
    write_varint(p.tf, out);
    prev = p.doc_id;
  }
  return Status::OK();
}


std::string FtsPostingCodec::EncodeSingle(uint32_t doc_id, uint32_t tf) {
  std::string buf;
  buf.reserve(6);
  write_varint(1, &buf);
  write_varint(doc_id, &buf);
  write_varint(tf, &buf);
  return buf;
}


Status FtsPostingCodec::Decode(const char *data, std::size_t size,
                               std::vector<FtsPosting> *out) {
  if (out == nullptr) {
    return Status::InvalidArgument("out is null");
  }
  out->clear();
  if (size == 0) {
    return Status::OK();
  }

  std::size_t pos = 0;
  uint32_t n = 0;
  if (!read_varint(data, size, &pos, &n)) {
    return Status::InvalidArgument("FtsPostingCodec::Decode: truncated count");
  }
  out->reserve(n);

  uint32_t prev = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t delta = 0;
    uint32_t tf = 0;
    if (!read_varint(data, size, &pos, &delta)) {
      return Status::InvalidArgument(
          "FtsPostingCodec::Decode: truncated doc_id delta");
    }
    if (!read_varint(data, size, &pos, &tf)) {
      return Status::InvalidArgument(
          "FtsPostingCodec::Decode: truncated term frequency");
    }
    if (i > 0 && delta == 0) {
      return Status::InvalidArgument(
          "FtsPostingCodec::Decode: duplicate doc_id in posting list");
    }
    prev += delta;
    out->push_back(FtsPosting{prev, tf});
  }
  if (pos != size) {
    return Status::InvalidArgument(
        "FtsPostingCodec::Decode: trailing bytes after posting list");
  }
  return Status::OK();
}


Status FtsPostingCodec::MergeEncoded(const char *lhs, std::size_t lhs_size,
                                     const char *rhs, std::size_t rhs_size,
                                     std::string *out) {
  if (out == nullptr) {
    return Status::InvalidArgument("out is null");
  }

  std::vector<FtsPosting> left;
  std::vector<FtsPosting> right;
  auto s = Decode(lhs, lhs_size, &left);
  if (!s.ok()) {
    return s;
  }
  s = Decode(rhs, rhs_size, &right);
  if (!s.ok()) {
    return s;
  }

  std::vector<FtsPosting> merged;
  merged.reserve(left.size() + right.size());
  std::size_t li = 0;
  std::size_t ri = 0;
  while (li < left.size() && ri < right.size()) {
    if (left[li].doc_id < right[ri].doc_id) {
      merged.push_back(left[li++]);
    } else if (right[ri].doc_id < left[li].doc_id) {
      merged.push_back(right[ri++]);
    } else {
      merged.push_back(FtsPosting{left[li].doc_id, left[li].tf + right[ri].tf});
      ++li;
      ++ri;
    }
  }
  while (li < left.size()) {
    merged.push_back(left[li++]);
  }
  while (ri < right.size()) {
    merged.push_back(right[ri++]);
  }
  return Encode(merged, out);
}


Status FtsPostingCodec::MergeSingleInto(std::vector<FtsPosting> *base,
                                        uint32_t doc_id, uint32_t tf) {
  if (base == nullptr) {
    return Status::InvalidArgument("base is null");
  }
  // Common case: the new doc_id is greater than the last one. Append.
  if (base->empty() || base->back().doc_id < doc_id) {
    base->push_back(FtsPosting{doc_id, tf});
    return Status::OK();
  }
  // Binary search for an insertion point / matching doc.
  auto it = std::lower_bound(
      base->begin(), base->end(), doc_id,
      [](const FtsPosting &p, uint32_t id) { return p.doc_id < id; });
  if (it != base->end() && it->doc_id == doc_id) {
    it->tf += tf;
  } else {
    base->insert(it, FtsPosting{doc_id, tf});
  }
  return Status::OK();
}


bool FtsPostingMerger::FullMergeV2(const MergeOperationInput &merge_in,
                                   MergeOperationOutput *merge_out) const {
  // Fast path: no existing value and a single operand — just copy through.
  if (merge_in.existing_value == nullptr &&
      merge_in.operand_list.size() == 1) {
    merge_out->new_value = std::string(merge_in.operand_list[0].data(),
                                       merge_in.operand_list[0].size());
    return true;
  }

  std::vector<FtsPosting> base;
  if (merge_in.existing_value != nullptr) {
    auto s = FtsPostingCodec::Decode(merge_in.existing_value->data(),
                                     merge_in.existing_value->size(), &base);
    if (!s.ok()) {
      LOG_ERROR("FtsPostingMerger: failed to decode existing value: %s",
                s.c_str());
      return false;
    }
  }

  std::vector<FtsPosting> operand_buf;
  for (const rocksdb::Slice &m : merge_in.operand_list) {
    auto s = FtsPostingCodec::Decode(m.data(), m.size(), &operand_buf);
    if (!s.ok()) {
      LOG_ERROR("FtsPostingMerger: failed to decode operand: %s", s.c_str());
      return false;
    }
    // Single-posting fast path (the overwhelmingly common insert operand).
    if (operand_buf.size() == 1) {
      auto ss = FtsPostingCodec::MergeSingleInto(&base, operand_buf[0].doc_id,
                                                 operand_buf[0].tf);
      if (!ss.ok()) {
        LOG_ERROR("FtsPostingMerger: single-posting merge failed: %s",
                  ss.c_str());
        return false;
      }
      continue;
    }
    // General case: N-way sorted merge into base.
    std::vector<FtsPosting> merged;
    merged.reserve(base.size() + operand_buf.size());
    std::size_t li = 0;
    std::size_t ri = 0;
    while (li < base.size() && ri < operand_buf.size()) {
      if (base[li].doc_id < operand_buf[ri].doc_id) {
        merged.push_back(base[li++]);
      } else if (operand_buf[ri].doc_id < base[li].doc_id) {
        merged.push_back(operand_buf[ri++]);
      } else {
        merged.push_back(FtsPosting{base[li].doc_id,
                                    base[li].tf + operand_buf[ri].tf});
        ++li;
        ++ri;
      }
    }
    while (li < base.size()) {
      merged.push_back(base[li++]);
    }
    while (ri < operand_buf.size()) {
      merged.push_back(operand_buf[ri++]);
    }
    base.swap(merged);
  }

  auto s = FtsPostingCodec::Encode(base, &merge_out->new_value);
  if (!s.ok()) {
    LOG_ERROR("FtsPostingMerger: failed to re-encode merged posting list: %s",
              s.c_str());
    return false;
  }
  return true;
}


bool FtsPostingMerger::PartialMerge(const rocksdb::Slice & /*key*/,
                                    const rocksdb::Slice &left_operand,
                                    const rocksdb::Slice &right_operand,
                                    std::string *new_value,
                                    rocksdb::Logger * /*logger*/) const {
  auto s = FtsPostingCodec::MergeEncoded(left_operand.data(),
                                         left_operand.size(),
                                         right_operand.data(),
                                         right_operand.size(), new_value);
  if (!s.ok()) {
    LOG_ERROR("FtsPostingMerger::PartialMerge failed: %s", s.c_str());
    return false;
  }
  return true;
}


const char *FtsPostingMerger::Name() const {
  return "FtsPostingMerger";
}


}  // namespace zvec
