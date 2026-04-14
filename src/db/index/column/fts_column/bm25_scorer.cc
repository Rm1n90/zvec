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


#include "bm25_scorer.h"

#include <cmath>


namespace zvec {


float Bm25Scorer::Idf(uint32_t df) const {
  // log( (N - df + 0.5) / (df + 0.5) + 1 )
  const double n = static_cast<double>(stats_.total_docs);
  const double d = static_cast<double>(df);
  // Guard against df > N (stale stats during concurrent writes): clamp the
  // numerator to zero, keeping log(... + 1) >= 0. When df == N the formula
  // still produces a small positive value (the classic Lucene behavior).
  const double num = n >= d ? (n - d + 0.5) : 0.0;
  const double den = d + 0.5;
  // den is always > 0 for non-negative df.
  return static_cast<float>(std::log(num / den + 1.0));
}


float Bm25Scorer::ScoreWithIdf(uint32_t tf, float idf, uint32_t doc_len) const {
  if (tf == 0) {
    return 0.0f;
  }
  const double avgdl = stats_.AvgDocLen();
  const double norm =
      (avgdl > 0.0)
          ? (1.0 - static_cast<double>(params_.b) +
             static_cast<double>(params_.b) *
                 (static_cast<double>(doc_len) / avgdl))
          : 1.0;
  const double k1 = static_cast<double>(params_.k1);
  const double tf_d = static_cast<double>(tf);
  const double numerator = tf_d * (k1 + 1.0);
  const double denominator = tf_d + k1 * norm;
  // denominator is always > 0: tf >= 1 and k1 * norm >= 0.
  return static_cast<float>(static_cast<double>(idf) * numerator / denominator);
}


float Bm25Scorer::Score(uint32_t tf, uint32_t df, uint32_t doc_len) const {
  return ScoreWithIdf(tf, Idf(df), doc_len);
}


}  // namespace zvec
