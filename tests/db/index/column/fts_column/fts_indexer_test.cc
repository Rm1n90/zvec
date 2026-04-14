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
#include <memory>
#include <string>
#include <vector>
#include <zvec/db/index_params.h>
#include <zvec/db/schema.h>
#include "db/index/column/fts_column/fts_indexer.h"
#include "tests/test_util.h"


using namespace zvec;


namespace {

constexpr const char *kCollectionName = "fts_test_collection";
constexpr const char *kWorkingDir = "./fts_indexer_test_dir/";
constexpr const char *kTextField = "body";
constexpr const char *kSecondField = "title";


FieldSchema MakeFtsField(const std::string &name) {
  return FieldSchema(name, DataType::STRING, /*nullable=*/true,
                     std::make_shared<FtsIndexParams>("default", 1.2f, 0.75f));
}


// Convert an FtsSearchResult into an ordered list of doc_ids (sorted by score
// descending — already the iteration order).
std::vector<uint32_t> ResultDocIds(const FtsSearchResult::Ptr &res) {
  std::vector<uint32_t> ids;
  if (!res) {
    return ids;
  }
  auto it = res->create_iterator();
  while (it->valid()) {
    ids.push_back(static_cast<uint32_t>(it->doc_id()));
    it->next();
  }
  return ids;
}


std::vector<float> ResultScores(const FtsSearchResult::Ptr &res) {
  std::vector<float> scores;
  if (!res) {
    return scores;
  }
  auto it = res->create_iterator();
  while (it->valid()) {
    scores.push_back(it->score());
    it->next();
  }
  return scores;
}

}  // namespace


class FtsIndexerTest : public testing::Test {
 protected:
  void SetUp() override {
    zvec::test_util::RemoveTestPath(kWorkingDir);
    indexer_ = FtsIndexer::CreateAndOpen(kCollectionName, kWorkingDir,
                                         /*create_dir_if_missing=*/true,
                                         {MakeFtsField(kTextField)},
                                         /*read_only=*/false);
    ASSERT_NE(indexer_, nullptr);
  }

  void TearDown() override {
    indexer_.reset();
    zvec::test_util::RemoveTestPath(kWorkingDir);
  }

  FtsColumnIndexer::Ptr column() {
    return (*indexer_)[kTextField];
  }

  FtsIndexer::Ptr indexer_;
};


TEST_F(FtsIndexerTest, OpenedIndexerExposesColumn) {
  ASSERT_NE(column(), nullptr);
  EXPECT_EQ(column()->TotalDocs(), 0u);
  EXPECT_EQ(column()->TotalTokens(), 0u);
  EXPECT_FALSE(column()->IsSealed());
}


TEST_F(FtsIndexerTest, InsertUpdatesStats) {
  auto col = column();
  ASSERT_TRUE(col->Insert(0, "the quick brown fox").ok());
  ASSERT_TRUE(col->Insert(1, "Hello World").ok());

  EXPECT_EQ(col->TotalDocs(), 2u);
  EXPECT_EQ(col->TotalTokens(), 6u);  // 4 + 2
  EXPECT_EQ(col->MaxId(), 1u);
}


TEST_F(FtsIndexerTest, SearchReturnsMatchingDoc) {
  auto col = column();
  ASSERT_TRUE(col->Insert(0, "the quick brown fox").ok());
  ASSERT_TRUE(col->Insert(1, "the lazy dog").ok());
  ASSERT_TRUE(col->Insert(2, "totally unrelated content").ok());

  auto res = col->Search("fox", 10);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0u);
}


TEST_F(FtsIndexerTest, SearchIsCaseInsensitive) {
  auto col = column();
  ASSERT_TRUE(col->Insert(0, "Hello World").ok());
  ASSERT_TRUE(col->Insert(1, "goodbye world").ok());

  auto res = col->Search("HELLO", 10);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0u);
}


TEST_F(FtsIndexerTest, OrSearchScoresMultiTermMatchHigher) {
  auto col = column();
  // Doc 0 contains both "quick" and "fox"; doc 1 contains just "quick";
  // doc 2 contains just "fox". With OR semantics, doc 0 should win.
  ASSERT_TRUE(col->Insert(0, "the quick brown fox jumps").ok());
  ASSERT_TRUE(col->Insert(1, "be quick about it").ok());
  ASSERT_TRUE(col->Insert(2, "a fox is a fox").ok());

  auto res = col->Search("quick fox", 10, FtsColumnIndexer::MatchOp::OR);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  auto scores = ResultScores(res.value());
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 0u);  // both terms present -> highest score
  EXPECT_GT(scores[0], scores[1]);
  EXPECT_GT(scores[0], scores[2]);
}


TEST_F(FtsIndexerTest, AndSearchRequiresAllTerms) {
  auto col = column();
  ASSERT_TRUE(col->Insert(0, "the quick brown fox").ok());
  ASSERT_TRUE(col->Insert(1, "be quick about it").ok());
  ASSERT_TRUE(col->Insert(2, "a fox is a fox").ok());

  auto res = col->Search("quick fox", 10, FtsColumnIndexer::MatchOp::AND);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0u);
}


TEST_F(FtsIndexerTest, AndSearchEmptyWhenAnyTermMissing) {
  auto col = column();
  ASSERT_TRUE(col->Insert(0, "alpha beta").ok());
  ASSERT_TRUE(col->Insert(1, "alpha gamma").ok());

  auto res = col->Search("alpha unknownterm", 10, FtsColumnIndexer::MatchOp::AND);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(ResultDocIds(res.value()).size(), 0u);
}


TEST_F(FtsIndexerTest, EmptyQueryReturnsEmpty) {
  auto col = column();
  ASSERT_TRUE(col->Insert(0, "anything").ok());

  auto res = col->Search("", 10);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value()->count(), 0u);
}


TEST_F(FtsIndexerTest, TopkLimitsResults) {
  auto col = column();
  for (uint32_t i = 0; i < 5; ++i) {
    ASSERT_TRUE(col->Insert(i, "common word here").ok());
  }
  auto res = col->Search("common", 3);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value()->count(), 3u);
}


TEST_F(FtsIndexerTest, TopkZeroReturnsEmpty) {
  auto col = column();
  ASSERT_TRUE(col->Insert(0, "alpha").ok());
  auto res = col->Search("alpha", 0);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value()->count(), 0u);
}


TEST_F(FtsIndexerTest, TermFrequencyInfluencesScore) {
  auto col = column();
  // doc 0 mentions "ranking" 3 times; doc 1 mentions it once.
  ASSERT_TRUE(col->Insert(0, "ranking ranking ranking and other words").ok());
  ASSERT_TRUE(col->Insert(1, "ranking is interesting").ok());

  auto res = col->Search("ranking", 10);
  ASSERT_TRUE(res.has_value());
  auto scores = ResultScores(res.value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 0u);
  EXPECT_GT(scores[0], scores[1]);
}


TEST_F(FtsIndexerTest, RareTermsScoreHigherThanCommonOnes) {
  auto col = column();
  // Insert 10 docs all containing "common", plus one extra rare doc.
  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_TRUE(col->Insert(i, "common word everywhere").ok());
  }
  ASSERT_TRUE(col->Insert(100, "rare unique sparkle").ok());

  // Query "common rare" — the doc with "rare" should score highest because
  // "rare" has much lower df and therefore higher IDF.
  auto res = col->Search("common rare", 5);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_GE(ids.size(), 1u);
  EXPECT_EQ(ids[0], 100u);
}


TEST_F(FtsIndexerTest, InsertNullCountsTowardDocsButNotTokens) {
  auto col = column();
  ASSERT_TRUE(col->Insert(0, "alpha beta").ok());
  ASSERT_TRUE(col->InsertNull(1).ok());
  ASSERT_TRUE(col->Insert(2, "gamma").ok());

  EXPECT_EQ(col->TotalDocs(), 3u);
  EXPECT_EQ(col->TotalTokens(), 3u);  // 2 + 0 + 1
  EXPECT_EQ(col->MaxId(), 2u);

  // The null doc should never appear in results.
  auto res = col->Search("alpha", 10);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0u);
}


TEST_F(FtsIndexerTest, SealMakesIndexerReadOnly) {
  auto col = column();
  ASSERT_TRUE(col->Insert(0, "before sealing").ok());
  ASSERT_TRUE(col->Seal().ok());
  EXPECT_TRUE(col->IsSealed());

  // Inserts must fail.
  auto s = col->Insert(1, "after sealing");
  EXPECT_FALSE(s.ok());

  // Search still works.
  auto res = col->Search("sealing", 10);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(ResultDocIds(res.value()).size(), 1u);
}


TEST_F(FtsIndexerTest, FlushAndReopenPreservesState) {
  {
    auto col = column();
    ASSERT_TRUE(col->Insert(0, "preserved across reopens").ok());
    ASSERT_TRUE(col->Insert(1, "another doc").ok());
    ASSERT_TRUE(indexer_->flush().ok());
  }

  // Drop and reopen the indexer.
  indexer_.reset();
  indexer_ = FtsIndexer::CreateAndOpen(kCollectionName, kWorkingDir,
                                       /*create_dir_if_missing=*/false,
                                       {MakeFtsField(kTextField)},
                                       /*read_only=*/false);
  ASSERT_NE(indexer_, nullptr);

  auto col = column();
  EXPECT_EQ(col->TotalDocs(), 2u);
  EXPECT_EQ(col->TotalTokens(), 5u);  // 3 + 2
  EXPECT_EQ(col->MaxId(), 1u);

  auto res = col->Search("preserved", 10);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0u);
}


TEST_F(FtsIndexerTest, SealedStatePersistsAcrossReopen) {
  {
    auto col = column();
    ASSERT_TRUE(col->Insert(0, "before sealing").ok());
    ASSERT_TRUE(col->Seal().ok());
  }
  indexer_.reset();
  indexer_ = FtsIndexer::CreateAndOpen(kCollectionName, kWorkingDir,
                                       /*create_dir_if_missing=*/false,
                                       {MakeFtsField(kTextField)},
                                       /*read_only=*/false);
  ASSERT_NE(indexer_, nullptr);
  EXPECT_TRUE(column()->IsSealed());
}


TEST_F(FtsIndexerTest, AddSecondColumnAtRuntime) {
  auto status = indexer_->create_column_indexer(MakeFtsField(kSecondField));
  ASSERT_TRUE(status.ok());
  ASSERT_NE((*indexer_)[kSecondField], nullptr);

  ASSERT_TRUE((*indexer_)[kTextField]->Insert(0, "body text").ok());
  ASSERT_TRUE((*indexer_)[kSecondField]->Insert(0, "title text").ok());

  // Each field's search is independent.
  auto body_res = (*indexer_)[kTextField]->Search("body", 10);
  ASSERT_TRUE(body_res.has_value());
  EXPECT_EQ(ResultDocIds(body_res.value()), std::vector<uint32_t>{0});

  auto title_res = (*indexer_)[kSecondField]->Search("title", 10);
  ASSERT_TRUE(title_res.has_value());
  EXPECT_EQ(ResultDocIds(title_res.value()), std::vector<uint32_t>{0});

  // "body" should not appear in the title field.
  auto crossed = (*indexer_)[kSecondField]->Search("body", 10);
  ASSERT_TRUE(crossed.has_value());
  EXPECT_EQ(crossed.value()->count(), 0u);
}


TEST_F(FtsIndexerTest, RemoveColumnDropsItsData) {
  ASSERT_TRUE(indexer_->create_column_indexer(MakeFtsField(kSecondField)).ok());
  ASSERT_TRUE((*indexer_)[kSecondField]->Insert(0, "to be dropped").ok());

  ASSERT_TRUE(indexer_->remove_column_indexer(kSecondField).ok());
  EXPECT_EQ((*indexer_)[kSecondField], nullptr);

  // The other column is untouched.
  ASSERT_NE((*indexer_)[kTextField], nullptr);
}


TEST_F(FtsIndexerTest, PunctuationSplitsTokensConsistently) {
  auto col = column();
  ASSERT_TRUE(col->Insert(0, "hello, world!").ok());

  auto res1 = col->Search("hello", 10);
  ASSERT_TRUE(res1.has_value());
  EXPECT_EQ(ResultDocIds(res1.value()).size(), 1u);

  auto res2 = col->Search("world", 10);
  ASSERT_TRUE(res2.has_value());
  EXPECT_EQ(ResultDocIds(res2.value()).size(), 1u);

  // Punctuation in the query is also stripped.
  auto res3 = col->Search("hello, world!", 10);
  ASSERT_TRUE(res3.has_value());
  EXPECT_EQ(ResultDocIds(res3.value()).size(), 1u);
}
