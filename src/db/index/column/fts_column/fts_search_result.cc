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


#include "fts_search_result.h"

#include "db/common/constants.h"
#include "db/common/typedef.h"


namespace zvec {


IndexResults::IteratorUPtr FtsSearchResult::create_iterator() {
  return std::make_unique<FtsIterator>(shared_from_this());
}


idx_t FtsSearchResult::FtsIterator::doc_id() const {
  if (!valid()) {
    return INVALID_DOC_ID;
  }
  return static_cast<idx_t>(owner_->hits_[pos_].doc_id);
}


float FtsSearchResult::FtsIterator::score() const {
  if (!valid()) {
    return 0.0f;
  }
  return owner_->hits_[pos_].score;
}


void FtsSearchResult::FtsIterator::next() {
  if (valid()) {
    ++pos_;
  }
}


bool FtsSearchResult::FtsIterator::valid() const {
  return owner_ && pos_ < owner_->hits_.size();
}


}  // namespace zvec
