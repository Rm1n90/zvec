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
// Tests for Phase 5 of the optimization pipeline:
//   - Sharded collections (write_shards_ > 1) create, round-trip, and
//     reopen correctly.
//   - Mixed-pk batches partition across every shard.
//   - Per-shard rotation is independent.
//   - Manifest back-compat: an N=1 manifest written by the new code is
//     readable, and an explicit write_shards_=4 persists across reopen.
//   - Delete batches partition by shard.
//
// These tests deliberately use small max_doc_count_per_segment (the
// minimum allowed, 1000) so per-shard rotation kicks in within realistic
// batch sizes.

#include <string>
#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/options.h>
#include "db/common/file_helper.h"
#include "db/common/pk_shard.h"
#include "index/utils/utils.h"

using namespace zvec;
using namespace zvec::test;

namespace {

const std::string kColPath = "test_collection_phase5";

class ShardedWritesPhase5Test : public ::testing::Test {
 protected:
  void SetUp() override { FileHelper::RemoveDirectory(kColPath); }
  void TearDown() override { FileHelper::RemoveDirectory(kColPath); }
};

CollectionOptions OptsWithShards(uint32_t n_shards) {
  CollectionOptions o;
  o.read_only_ = false;
  o.enable_mmap_ = true;
  o.max_buffer_size_ = 64 * 1024 * 1024;
  o.write_shards_ = n_shards;
  return o;
}

}  // namespace

// --- Smoke: N=1 collection still works exactly as before ------------------
TEST_F(ShardedWritesPhase5Test, UnshardedCollectionStillWorks) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = OptsWithShards(1);
  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  std::vector<Doc> docs;
  for (int i = 0; i < 50; ++i) docs.push_back(TestHelper::CreateDoc(i, *schema));
  auto r = collection->Insert(docs);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  EXPECT_EQ(r.value().size(), 50u);
  for (const auto &s : r.value()) EXPECT_TRUE(s.ok()) << s.message();

  // Reopen — read back all 50.
  collection.reset();
  auto reopened = Collection::Open(kColPath, opts);
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto stats = reopened.value()->Stats().value();
  EXPECT_EQ(stats.doc_count, 50u);
}

// --- N=4: every pk is readable after insert; every shard got some data ----
TEST_F(ShardedWritesPhase5Test, ShardedInsertDistributesAcrossShards) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = OptsWithShards(4);
  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  constexpr int kDocs = 200;
  std::vector<Doc> docs;
  docs.reserve(kDocs);
  for (int i = 0; i < kDocs; ++i) {
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  }

  auto r = collection->Insert(docs);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  ASSERT_EQ(r.value().size(), static_cast<size_t>(kDocs));
  for (int i = 0; i < kDocs; ++i) {
    EXPECT_TRUE(r.value()[i].ok()) << "doc i=" << i;
  }

  // Every pk must be fetchable across all shards (Fetch uses
  // get_all_segments which fans out via all_writing_segments()).
  for (int i = 0; i < kDocs; ++i) {
    auto fetched = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched.value().size(), 1u) << "missing pk_" << i;
  }

  // Sanity check: routing actually spreads pks across all 4 shards.
  std::unordered_set<size_t> shard_hits;
  for (int i = 0; i < kDocs; ++i) {
    shard_hits.insert(PkToShard(TestHelper::MakePK(i), 4));
  }
  EXPECT_EQ(shard_hits.size(), 4u) << "workload should touch every shard";

  auto stats = collection->Stats().value();
  EXPECT_EQ(stats.doc_count, static_cast<uint64_t>(kDocs));
}

// --- N=4 collection survives close + reopen with same write_shards --------
TEST_F(ShardedWritesPhase5Test, ShardedCollectionReopenPreservesShards) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = OptsWithShards(4);

  {
    auto collection = TestHelper::CreateCollectionWithDoc(
        kColPath, *schema, opts, 0, 0, false);
    ASSERT_NE(collection, nullptr);

    std::vector<Doc> docs;
    for (int i = 0; i < 100; ++i) docs.push_back(TestHelper::CreateDoc(i, *schema));
    auto r = collection->Insert(docs);
    ASSERT_TRUE(r.has_value()) << r.error().message();

    auto stats = collection->Stats().value();
    EXPECT_EQ(stats.doc_count, 100u);
  }

  // Reopen WITHOUT passing write_shards — should still load N=4 from
  // the manifest. The in-memory options_.write_shards_ is ignored at
  // Open time; manifest is authoritative.
  CollectionOptions reopen_opts;
  reopen_opts.read_only_ = false;
  reopen_opts.enable_mmap_ = true;
  reopen_opts.max_buffer_size_ = 64 * 1024 * 1024;
  reopen_opts.write_shards_ = 1;  // deliberately "wrong" — should be ignored

  auto reopened = Collection::Open(kColPath, reopen_opts);
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto stats = reopened.value()->Stats().value();
  EXPECT_EQ(stats.doc_count, 100u);

  // Every pk must still be fetchable.
  for (int i = 0; i < 100; ++i) {
    auto fetched =
        reopened.value()->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched.value().size(), 1u) << "missing pk_" << i << " after reopen";
  }
}

// --- N=4 per-shard rotation: push one shard past max_doc_count ------------
//
// With max_doc_count_per_segment = 1000 and a large pk range, the
// shard-0 writing segment will rotate independently of the others once
// it fills. The collection must still expose every inserted pk.
TEST_F(ShardedWritesPhase5Test, PerShardRotationIsIndependent) {
  // Schema with the minimum allowed max_doc_count_per_segment (1000).
  auto schema = TestHelper::CreateNormalSchema(
      /*nullable=*/false, /*name=*/"demo", /*scalar_index_params=*/nullptr,
      std::make_shared<HnswIndexParams>(MetricType::IP),
      /*max_doc_count_per_segment=*/1000);
  auto opts = OptsWithShards(4);
  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  // With random-ish hashing, each shard gets ~N/4 docs, so ~4000 pks
  // will saturate any single shard at 1000 docs and force rotation.
  constexpr int kDocs = 4000;
  std::vector<Doc> docs;
  docs.reserve(kDocs);
  for (int i = 0; i < kDocs; ++i) {
    docs.push_back(TestHelper::CreateDoc(i, *schema));
  }
  // CollectionImpl::write_impl caps incoming batch at kMaxWriteBatchSize
  // (1024), so split the insert to stay under the limit.
  constexpr int kChunk = 1000;
  for (int start = 0; start < kDocs; start += kChunk) {
    int end = std::min(start + kChunk, kDocs);
    std::vector<Doc> chunk(docs.begin() + start, docs.begin() + end);
    auto r = collection->Insert(chunk);
    ASSERT_TRUE(r.has_value()) << r.error().message();
    for (const auto &s : r.value()) EXPECT_TRUE(s.ok()) << s.message();
  }

  auto stats = collection->Stats().value();
  EXPECT_EQ(stats.doc_count, static_cast<uint64_t>(kDocs));

  // Spot-check distribution end-to-end: every pk fetchable.
  for (int i : {0, 500, 1000, 1500, 2000, 2500, 3000, 3999}) {
    auto fetched = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched.value().size(), 1u) << "missing pk_" << i;
  }
}

// --- N=4: UPSERT tombstones existing across shards ------------------------
TEST_F(ShardedWritesPhase5Test, ShardedUpsertRewritesPks) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = OptsWithShards(4);
  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  // First insert 30 docs across all shards.
  std::vector<Doc> first;
  for (int i = 0; i < 30; ++i) first.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->Insert(first).has_value());

  // Upsert: 10 existing (0-9) + 10 new (30-39). Mix spans every shard.
  std::vector<Doc> upsert;
  for (int i = 0; i < 10; ++i) upsert.push_back(TestHelper::CreateDoc(i, *schema));
  for (int i = 30; i < 40; ++i) upsert.push_back(TestHelper::CreateDoc(i, *schema));
  auto r = collection->Upsert(upsert);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  for (const auto &s : r.value()) EXPECT_TRUE(s.ok()) << s.message();

  // Should now have 40 distinct pks (0..39).
  auto stats = collection->Stats().value();
  EXPECT_EQ(stats.doc_count, 40u);
  for (int i = 0; i < 40; ++i) {
    auto fetched = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched.value().size(), 1u) << "missing pk_" << i;
  }
}

// --- N=4: Delete partitions pks across shards correctly -------------------
TEST_F(ShardedWritesPhase5Test, ShardedDeletePartitionsByShard) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = OptsWithShards(4);
  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  std::vector<Doc> docs;
  for (int i = 0; i < 50; ++i) docs.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->Insert(docs).has_value());

  // Delete a subset that spans every shard.
  std::vector<std::string> pks;
  for (int i = 0; i < 50; i += 3) pks.push_back(TestHelper::MakePK(i));
  auto r = collection->Delete(pks);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  EXPECT_EQ(r.value().size(), pks.size());
  for (const auto &s : r.value()) EXPECT_TRUE(s.ok()) << s.message();

  // Deleted pks return nullptr (Fetch always puts the key in the map
  // with nullptr if the pk is missing or tombstoned); kept pks fetch a
  // non-null Doc pointer.
  for (int i = 0; i < 50; ++i) {
    auto fetched = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(fetched.has_value());
    auto it = fetched.value().find(TestHelper::MakePK(i));
    ASSERT_NE(it, fetched.value().end());
    if (i % 3 == 0) {
      EXPECT_EQ(it->second, nullptr) << "pk_" << i << " should be gone";
    } else {
      EXPECT_NE(it->second, nullptr) << "pk_" << i << " wrongly missing";
    }
  }
}
