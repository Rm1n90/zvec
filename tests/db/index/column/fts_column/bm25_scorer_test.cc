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


#include <cmath>
#include <gtest/gtest.h>
#include "db/index/column/fts_column/bm25_scorer.h"


using namespace zvec;


namespace {

Bm25CollectionStats MakeStats(uint64_t total_docs, uint64_t total_tokens) {
  Bm25CollectionStats s;
  s.total_docs = total_docs;
  s.total_tokens = total_tokens;
  return s;
}

}  // namespace


TEST(Bm25CollectionStatsTest, AvgDocLenZeroWhenEmpty) {
  Bm25CollectionStats s;
  EXPECT_DOUBLE_EQ(s.AvgDocLen(), 0.0);
}


TEST(Bm25CollectionStatsTest, AvgDocLenComputesRatio) {
  auto s = MakeStats(10, 80);
  EXPECT_DOUBLE_EQ(s.AvgDocLen(), 8.0);
}


TEST(Bm25ScorerTest, IdfIsNonNegativeAcrossValidRange) {
  auto stats = MakeStats(100, 1000);
  Bm25Scorer scorer({}, stats);
  for (uint32_t df = 0; df <= 100; ++df) {
    EXPECT_GE(scorer.Idf(df), 0.0f) << "df=" << df;
  }
}


TEST(Bm25ScorerTest, IdfDecreasesAsDfIncreases) {
  auto stats = MakeStats(1000, 10000);
  Bm25Scorer scorer({}, stats);
  float prev = scorer.Idf(1);
  for (uint32_t df = 2; df <= 500; ++df) {
    float cur = scorer.Idf(df);
    ASSERT_LE(cur, prev) << "df=" << df;
    prev = cur;
  }
}


TEST(Bm25ScorerTest, IdfClampsWhenDfExceedsCorpus) {
  auto stats = MakeStats(10, 100);
  Bm25Scorer scorer({}, stats);
  // A stale stat where df>N would otherwise produce log of a negative
  // number; we clamp the numerator to zero and thus return log(1) = 0.
  EXPECT_FLOAT_EQ(scorer.Idf(20), 0.0f);
}


TEST(Bm25ScorerTest, ScoreIsZeroWhenTfIsZero) {
  auto stats = MakeStats(10, 80);
  Bm25Scorer scorer({}, stats);
  EXPECT_FLOAT_EQ(scorer.Score(0, 2, 10), 0.0f);
}


TEST(Bm25ScorerTest, ScoreIncreasesWithTermFrequency) {
  auto stats = MakeStats(10, 80);
  Bm25Scorer scorer({}, stats);
  float prev = 0.0f;
  for (uint32_t tf = 1; tf <= 20; ++tf) {
    float cur = scorer.Score(tf, 2, 10);
    ASSERT_GT(cur, prev) << "tf=" << tf;
    prev = cur;
  }
}


TEST(Bm25ScorerTest, ScoreDecreasesAsDfIncreases) {
  auto stats = MakeStats(1000, 10000);
  Bm25Scorer scorer({}, stats);
  float prev = scorer.Score(3, 1, 10);
  for (uint32_t df = 2; df <= 500; ++df) {
    float cur = scorer.Score(3, df, 10);
    ASSERT_LE(cur, prev) << "df=" << df;
    prev = cur;
  }
}


TEST(Bm25ScorerTest, ScoreDecreasesWithLongerDocWhenBPositive) {
  auto stats = MakeStats(100, 1000);  // avgdl = 10
  Bm25Scorer scorer({}, stats);       // b = 0.75 default
  float short_doc = scorer.Score(3, 2, 5);
  float avg_doc = scorer.Score(3, 2, 10);
  float long_doc = scorer.Score(3, 2, 40);
  EXPECT_GT(short_doc, avg_doc);
  EXPECT_GT(avg_doc, long_doc);
}


TEST(Bm25ScorerTest, ScoreIgnoresDocLengthWhenBIsZero) {
  Bm25Params params;
  params.b = 0.0f;
  auto stats = MakeStats(100, 1000);
  Bm25Scorer scorer(params, stats);
  float s1 = scorer.Score(3, 2, 1);
  float s2 = scorer.Score(3, 2, 10);
  float s3 = scorer.Score(3, 2, 1000);
  EXPECT_FLOAT_EQ(s1, s2);
  EXPECT_FLOAT_EQ(s2, s3);
}


TEST(Bm25ScorerTest, MatchesManualComputation) {
  // N = 10, total_tokens = 80 -> avgdl = 8.
  // tf = 3, df = 2, doc_len = 10, k1 = 1.2, b = 0.75.
  //
  //   idf  = ln( (10 - 2 + 0.5) / (2 + 0.5) + 1 )
  //        = ln(8.5 / 2.5 + 1)
  //        = ln(4.4)
  //   norm = 1 - 0.75 + 0.75 * (10 / 8) = 1.1875
  //   score = idf * 3 * 2.2 / (3 + 1.2 * 1.1875)
  //         = idf * 6.6 / 4.425
  auto stats = MakeStats(10, 80);
  Bm25Scorer scorer({}, stats);

  const double expected_idf = std::log(4.4);
  EXPECT_NEAR(scorer.Idf(2), static_cast<float>(expected_idf), 1e-5f);

  const double expected_score = expected_idf * 6.6 / 4.425;
  EXPECT_NEAR(scorer.Score(3, 2, 10), static_cast<float>(expected_score),
              1e-5f);
}


TEST(Bm25ScorerTest, ScoreWithIdfMatchesFullScore) {
  auto stats = MakeStats(50, 500);
  Bm25Scorer scorer({}, stats);
  for (uint32_t df : {1u, 5u, 25u, 49u}) {
    const float idf = scorer.Idf(df);
    for (uint32_t tf : {1u, 3u, 10u}) {
      for (uint32_t len : {1u, 10u, 100u}) {
        EXPECT_FLOAT_EQ(scorer.ScoreWithIdf(tf, idf, len),
                        scorer.Score(tf, df, len))
            << "df=" << df << " tf=" << tf << " len=" << len;
      }
    }
  }
}


TEST(Bm25ScorerTest, EmptyCollectionDoesNotCrash) {
  Bm25CollectionStats stats;  // total_docs = 0, total_tokens = 0
  Bm25Scorer scorer({}, stats);
  // No docs means avgdl = 0; scorer should still produce a finite value.
  float s = scorer.Score(3, 0, 5);
  EXPECT_TRUE(std::isfinite(s));
}
