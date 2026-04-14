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
#include <zvec/ailego/buffer/buffer_manager.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/schema.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "db/index/column/fts_column/fts_column_indexer.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/version_manager.h"
#include "db/index/segment/segment.h"
#include "tests/test_util.h"


using namespace zvec;


namespace {

constexpr const char *kColName = "test_segment_fts";
constexpr const char *kColPath = "./test_segment_fts_dir";
constexpr const char *kFtsField = "body";


CollectionSchema::Ptr MakeFtsSchema() {
  auto schema = std::make_shared<CollectionSchema>(kColName);
  auto fts_params =
      std::make_shared<FtsIndexParams>("default", 1.2f, 0.75f);
  schema->add_field(std::make_shared<FieldSchema>(
      kFtsField, DataType::STRING, /*nullable=*/true, fts_params));
  return schema;
}


Doc MakeDoc(const std::string &pk, const std::string &body) {
  Doc doc;
  doc.set_pk(pk);
  doc.set<std::string>(kFtsField, body);
  return doc;
}


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

}  // namespace


class SegmentFtsTest : public testing::Test {
 protected:
  void SetUp() override {
    zvec::test_util::RemoveTestPath(kColPath);
    ASSERT_TRUE(FileHelper::CreateDirectory(kColPath));

    ailego::BufferManager::Instance().init(MIN_MEMORY_LIMIT_BYTES, 1);

    schema_ = MakeFtsSchema();

    auto idmap_path = FileHelper::MakeFilePath(kColPath, FileID::ID_FILE, 0);
    id_map_ = IDMap::CreateAndOpen(kColName, idmap_path,
                                   /*create_dir_if_missing=*/true,
                                   /*read_only=*/false);
    ASSERT_NE(id_map_, nullptr);

    delete_store_ = std::make_shared<DeleteStore>(kColName);

    Version version;
    version.set_schema(*schema_);
    version.set_enable_mmap(false);
    auto vm_result = VersionManager::Create(kColPath, version);
    ASSERT_TRUE(vm_result.has_value());
    version_manager_ = vm_result.value();

    options_.read_only_ = false;
    options_.enable_mmap_ = false;
    options_.max_buffer_size_ = 64 * 1024 * 1024;
  }

  void TearDown() override {
    id_map_.reset();
    delete_store_.reset();
    version_manager_.reset();
    zvec::test_util::RemoveTestPath(kColPath);
  }

  Segment::Ptr CreateFreshSegment() {
    auto result = Segment::CreateAndOpen(kColPath, *schema_,
                                         /*segment_id=*/0,
                                         /*min_doc_id=*/0, id_map_,
                                         delete_store_, version_manager_,
                                         options_);
    if (!result.has_value()) {
      return nullptr;
    }
    return result.value();
  }

  Segment::Ptr ReopenSegment() {
    auto v = version_manager_->get_current_version();
    auto result = Segment::Open(kColPath, *schema_, *v.writing_segment_meta(),
                                id_map_, delete_store_, version_manager_,
                                options_);
    if (!result.has_value()) {
      return nullptr;
    }
    return result.value();
  }

  CollectionSchema::Ptr schema_;
  IDMap::Ptr id_map_;
  DeleteStore::Ptr delete_store_;
  VersionManager::Ptr version_manager_;
  SegmentOptions options_{};
};


TEST_F(SegmentFtsTest, FreshSegmentExposesFtsIndexer) {
  auto segment = CreateFreshSegment();
  ASSERT_NE(segment, nullptr);

  auto fts = segment->get_fts_indexer(kFtsField);
  ASSERT_NE(fts, nullptr);
  EXPECT_EQ(fts->TotalDocs(), 0u);
}


TEST_F(SegmentFtsTest, InsertingDocsRoutesTextIntoFtsIndexer) {
  auto segment = CreateFreshSegment();
  ASSERT_NE(segment, nullptr);

  std::vector<std::string> bodies{
      "the quick brown fox jumps over the lazy dog",
      "lorem ipsum dolor sit amet",
      "another quick example for indexing",
  };
  for (std::size_t i = 0; i < bodies.size(); ++i) {
    Doc d = MakeDoc("doc_" + std::to_string(i), bodies[i]);
    ASSERT_TRUE(segment->Insert(d).ok());
  }

  auto fts = segment->get_fts_indexer(kFtsField);
  ASSERT_NE(fts, nullptr);
  EXPECT_EQ(fts->TotalDocs(), 3u);

  auto res = fts->Search("quick", 10);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 2u);
  // The two matching docs are doc 0 and doc 2 (segment-local ids).
  EXPECT_TRUE((ids[0] == 0u && ids[1] == 2u) ||
              (ids[0] == 2u && ids[1] == 0u));
}


TEST_F(SegmentFtsTest, NullValueDoesNotIndexButCounts) {
  auto segment = CreateFreshSegment();
  ASSERT_NE(segment, nullptr);

  Doc d0 = MakeDoc("a", "alpha bravo");
  Doc d1;
  d1.set_pk("b");  // no FTS field set => null
  Doc d2 = MakeDoc("c", "charlie alpha");

  ASSERT_TRUE(segment->Insert(d0).ok());
  ASSERT_TRUE(segment->Insert(d1).ok());
  ASSERT_TRUE(segment->Insert(d2).ok());

  auto fts = segment->get_fts_indexer(kFtsField);
  ASSERT_NE(fts, nullptr);
  EXPECT_EQ(fts->TotalDocs(), 3u);
  // alpha appears in 2 docs, bravo in 1, charlie in 1; total tokens =
  // 2 + 0 + 2 = 4.
  EXPECT_EQ(fts->TotalTokens(), 4u);

  auto res = fts->Search("alpha", 10);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_TRUE((ids[0] == 0u && ids[1] == 2u) ||
              (ids[0] == 2u && ids[1] == 0u));
}


TEST_F(SegmentFtsTest, ReopenSegmentPreservesFtsIndex) {
  {
    auto segment = CreateFreshSegment();
    ASSERT_NE(segment, nullptr);

    Doc a = MakeDoc("a", "preserve me across reopens");
    Doc b = MakeDoc("b", "another doc");
    ASSERT_TRUE(segment->Insert(a).ok());
    ASSERT_TRUE(segment->Insert(b).ok());

    ASSERT_TRUE(segment->flush().ok());
  }

  auto segment = ReopenSegment();
  ASSERT_NE(segment, nullptr);

  auto fts = segment->get_fts_indexer(kFtsField);
  ASSERT_NE(fts, nullptr);
  EXPECT_EQ(fts->TotalDocs(), 2u);

  auto res = fts->Search("preserve", 10);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0u);
}


TEST_F(SegmentFtsTest, DumpSealsFtsIndexer) {
  auto segment = CreateFreshSegment();
  ASSERT_NE(segment, nullptr);

  Doc d = MakeDoc("a", "alpha bravo charlie");
  ASSERT_TRUE(segment->Insert(d).ok());

  ASSERT_TRUE(segment->dump().ok());

  auto fts = segment->get_fts_indexer(kFtsField);
  ASSERT_NE(fts, nullptr);
  EXPECT_TRUE(fts->IsSealed());

  // Search still works on a sealed indexer.
  auto res = fts->Search("alpha", 10);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0u);
}


TEST_F(SegmentFtsTest, ScoringRanksMoreRelevantDocsHigher) {
  auto segment = CreateFreshSegment();
  ASSERT_NE(segment, nullptr);

  // doc 0 has "ranking" three times, doc 1 has it once.
  Doc many = MakeDoc("a", "ranking ranking ranking is great");
  Doc once = MakeDoc("b", "a single mention of ranking");
  ASSERT_TRUE(segment->Insert(many).ok());
  ASSERT_TRUE(segment->Insert(once).ok());

  auto fts = segment->get_fts_indexer(kFtsField);
  ASSERT_NE(fts, nullptr);
  auto res = fts->Search("ranking", 10);
  ASSERT_TRUE(res.has_value());
  auto ids = ResultDocIds(res.value());
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 0u);
  EXPECT_EQ(ids[1], 1u);
}
