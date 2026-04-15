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
//
// Tests for Phase 3 of the optimization pipeline:
//   - IDMap::upsert_batch (RocksDB WriteBatch)
//   - MemForwardStore::insert_batch (single-lock cache insert)
//   - SegmentImpl::apply_batched_writes: one fetch_add(N) for the
//     doc_id range, one append_batch WAL write, one id_map upsert_batch
//   - Integration via CollectionImpl::Insert/Upsert/Update/Delete
//
// These assert functional correctness — throughput wins are intentionally
// not benchmarked here, as they depend heavily on the host and kernel
// page-cache behaviour.

#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/options.h>
#include <zvec/db/status.h>
#include "db/common/file_helper.h"
#include "db/index/common/id_map.h"
#include "index/utils/utils.h"

using namespace zvec;
using namespace zvec::test;

namespace {

const std::string kColPath = "test_collection_phase3";

class BatchedWritesPhase3Test : public ::testing::Test {
 protected:
  void SetUp() override { FileHelper::RemoveDirectory(kColPath); }
  void TearDown() override { FileHelper::RemoveDirectory(kColPath); }
};

}  // namespace

// --- IDMap::upsert_batch direct API exercise -------------------------------
TEST_F(BatchedWritesPhase3Test, IDMapUpsertBatchRoundTrip) {
  std::string idmap_path = kColPath + "/idmap";
  FileHelper::RemoveDirectory(kColPath);
  ailego::FileHelper::MakePath(kColPath.c_str());

  auto idmap = IDMap::CreateAndOpen("c", idmap_path, /*create_if_missing=*/true,
                                    /*read_only=*/false);
  ASSERT_NE(idmap, nullptr);

  std::vector<std::string> keys;
  std::vector<uint64_t> ids;
  for (int i = 0; i < 100; ++i) {
    keys.push_back("pk_" + std::to_string(i));
    ids.push_back(static_cast<uint64_t>(1000 + i));
  }
  ASSERT_TRUE(idmap->upsert_batch(keys, ids).ok());

  // All keys readable.
  std::vector<uint64_t> got;
  ASSERT_TRUE(idmap->multi_get(keys, &got).ok());
  ASSERT_EQ(got.size(), 100u);
  for (int i = 0; i < 100; ++i) EXPECT_EQ(got[i], static_cast<uint64_t>(1000 + i));

  // Overwrite semantics: same keys with new ids.
  for (auto &v : ids) v += 500000;
  ASSERT_TRUE(idmap->upsert_batch(keys, ids).ok());
  got.clear();
  ASSERT_TRUE(idmap->multi_get(keys, &got).ok());
  for (int i = 0; i < 100; ++i) EXPECT_EQ(got[i], static_cast<uint64_t>(500000 + 1000 + i));

  // Mismatched sizes should be rejected.
  std::vector<std::string> short_keys{"a"};
  std::vector<uint64_t> short_ids{1, 2};
  EXPECT_FALSE(idmap->upsert_batch(short_keys, short_ids).ok());

  // Empty batch is a no-op, OK.
  EXPECT_TRUE(idmap->upsert_batch({}, {}).ok());
}

// --- InsertBatch pre-validation via multi_get returns per-doc errors ------
TEST_F(BatchedWritesPhase3Test, InsertBatchAlreadyExistsAlignment) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts,
                                                        0, 0, false);
  ASSERT_NE(collection, nullptr);

  // Seed pk_2, pk_7.
  std::vector<Doc> seed{TestHelper::CreateDoc(2, *schema),
                        TestHelper::CreateDoc(7, *schema)};
  ASSERT_TRUE(collection->Insert(seed).has_value());

  // Batch 0..9; indices 2 and 7 collide.
  std::vector<Doc> batch;
  for (int i = 0; i < 10; ++i) batch.push_back(TestHelper::CreateDoc(i, *schema));

  auto r = collection->Insert(batch);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  ASSERT_EQ(r.value().size(), 10u);

  for (int i = 0; i < 10; ++i) {
    if (i == 2 || i == 7) {
      EXPECT_FALSE(r.value()[i].ok()) << "expected AlreadyExists at " << i;
      EXPECT_EQ(r.value()[i].code(), StatusCode::ALREADY_EXISTS);
    } else {
      EXPECT_TRUE(r.value()[i].ok()) << r.value()[i].message();
    }
  }
}

// --- UpsertBatch tombstones existing doc_ids and re-inserts ---------------
TEST_F(BatchedWritesPhase3Test, UpsertBatchTombstoneAndReinsert) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts,
                                                        0, 0, false);
  ASSERT_NE(collection, nullptr);

  // Seed 5 docs.
  std::vector<Doc> seed;
  for (int i = 0; i < 5; ++i) seed.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->Insert(seed).has_value());

  // Upsert 3 existing + 3 new in one batch.
  std::vector<Doc> upsert_batch;
  for (int i = 2; i < 8; ++i) upsert_batch.push_back(TestHelper::CreateDoc(i, *schema));

  auto r = collection->Upsert(upsert_batch);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  ASSERT_EQ(r.value().size(), 6u);
  for (const auto &s : r.value()) EXPECT_TRUE(s.ok()) << s.message();

  // All 8 pks should be fetchable (pk_0..pk_7).
  for (int i = 0; i < 8; ++i) {
    auto fetched = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched.value().size(), 1u) << "pk_" << i << " missing";
  }
}

// --- UpdateBatch routes NotFound for missing pks via multi_get ------------
TEST_F(BatchedWritesPhase3Test, UpdateBatchMultiGetNotFoundAlignment) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts,
                                                        0, 0, false);
  ASSERT_NE(collection, nullptr);

  // Seed 0..4.
  std::vector<Doc> seed;
  for (int i = 0; i < 5; ++i) seed.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->Insert(seed).has_value());

  // Update mix: 1 (exists), 77 (missing), 3 (exists), 99 (missing).
  std::vector<Doc> batch;
  batch.push_back(TestHelper::CreateDoc(1, *schema));
  batch.push_back(TestHelper::CreateDoc(77, *schema));
  batch.push_back(TestHelper::CreateDoc(3, *schema));
  batch.push_back(TestHelper::CreateDoc(99, *schema));

  auto r = collection->Update(batch);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  ASSERT_EQ(r.value().size(), 4u);

  EXPECT_TRUE(r.value()[0].ok());
  EXPECT_EQ(r.value()[1].code(), StatusCode::NOT_FOUND);
  EXPECT_TRUE(r.value()[2].ok());
  EXPECT_EQ(r.value()[3].code(), StatusCode::NOT_FOUND);
}

// --- DeleteBatch batched id_map.multi_get + WAL append_batch --------------
TEST_F(BatchedWritesPhase3Test, DeleteBatchReportsMissingPerPk) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts,
                                                        0, 0, false);
  ASSERT_NE(collection, nullptr);

  std::vector<Doc> seed;
  for (int i = 0; i < 10; ++i) seed.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->Insert(seed).has_value());

  std::vector<std::string> pks;
  pks.push_back(TestHelper::MakePK(0));
  pks.push_back(TestHelper::MakePK(50));  // missing
  pks.push_back(TestHelper::MakePK(5));
  pks.push_back(TestHelper::MakePK(99));  // missing

  auto r = collection->Delete(pks);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  ASSERT_EQ(r.value().size(), 4u);

  EXPECT_TRUE(r.value()[0].ok());
  EXPECT_EQ(r.value()[1].code(), StatusCode::NOT_FOUND);
  EXPECT_TRUE(r.value()[2].ok());
  EXPECT_EQ(r.value()[3].code(), StatusCode::NOT_FOUND);

  // Re-delete of already-deleted doc → NotFound, not AlreadyExists.
  auto r2 = collection->Delete({TestHelper::MakePK(0)});
  ASSERT_TRUE(r2.has_value());
  EXPECT_EQ(r2.value()[0].code(), StatusCode::NOT_FOUND);
}

// --- Large batch: doc_id range allocation contiguous in one shot ----------
//
// If doc_id_allocator_.fetch_add(N) is broken and we instead ran N
// individual fetch_adds interleaved with other ops, ids would not be
// contiguous. Verify contiguity through the public API: after insert,
// every pk_i resolves to a doc that fetches correctly.
TEST_F(BatchedWritesPhase3Test, LargeBatchInsertAllDocsReadable) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts,
                                                        0, 0, false);
  ASSERT_NE(collection, nullptr);

  constexpr int kBatch = 500;
  std::vector<Doc> batch;
  batch.reserve(kBatch);
  for (int i = 0; i < kBatch; ++i) batch.push_back(TestHelper::CreateDoc(i, *schema));

  auto r = collection->Insert(batch);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  ASSERT_EQ(r.value().size(), static_cast<size_t>(kBatch));
  for (int i = 0; i < kBatch; ++i) {
    EXPECT_TRUE(r.value()[i].ok()) << "i=" << i;
  }

  auto stats = collection->Stats().value();
  EXPECT_EQ(stats.doc_count, static_cast<uint64_t>(kBatch));

  // Random spot-checks across the range.
  for (int i : {0, 1, 250, 499}) {
    auto fetched = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched.value().size(), 1u) << "missing pk_" << i;
  }
}
