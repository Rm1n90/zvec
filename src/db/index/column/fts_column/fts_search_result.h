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
#include <memory>
#include <vector>
#include "db/index/column/common/index_results.h"


namespace zvec {


/**
 * @brief A scored hit produced by an FTS search.
 *
 * `score` is the BM25 score (higher = more relevant).
 */
struct FtsScoredHit {
  uint32_t doc_id{0};
  float score{0.0f};
};


/**
 * @brief Top-K result of an FTS search.
 *
 * Hits are stored sorted by descending score (and ascending doc_id as a
 * tie-break) so that iteration yields the most relevant docs first.
 *
 * Unlike the inverted-index result, each hit carries a real BM25 score —
 * downstream code that pipes results through @ref IndexResults can rely on
 * the iterator's `score()` method returning the BM25 value.
 */
class FtsSearchResult : public IndexResults,
                        public std::enable_shared_from_this<FtsSearchResult> {
 public:
  using Ptr = std::shared_ptr<FtsSearchResult>;

  /// Construct from a vector of hits already sorted descending by score.
  /// If @p hits is not sorted, results from iteration are undefined.
  explicit FtsSearchResult(std::vector<FtsScoredHit> hits)
      : hits_(std::move(hits)) {}

  std::size_t count() const override {
    return hits_.size();
  }

  IteratorUPtr create_iterator() override;

  const std::vector<FtsScoredHit> &hits() const {
    return hits_;
  }

 private:
  class FtsIterator : public Iterator {
   public:
    explicit FtsIterator(std::shared_ptr<const FtsSearchResult> owner)
        : owner_(std::move(owner)) {}

    idx_t doc_id() const override;
    float score() const override;
    void next() override;
    bool valid() const override;

   private:
    std::shared_ptr<const FtsSearchResult> owner_;
    std::size_t pos_{0};
  };

  std::vector<FtsScoredHit> hits_;
};


}  // namespace zvec
