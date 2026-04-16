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
// Tests for Phase 5.5 of the optimization pipeline:
//   - Per-shard write locks allow multiple writer threads to insert
//     concurrently into different shards without blocking each other.
//   - write_mtx_ is held SHARED during writes so DDL / Optimize can
//     still acquire it exclusively and quiesce all writers.
//   - VersionManager::modify_and_apply prevents the read-modify-write
//     race when two shards rotate concurrently.
//
// These tests assert CORRECTNESS under concurrency — no data loss, no
// duplicate inserts, no crashes. Throughput scaling is not asserted
// because it depends on hardware and OS scheduling.

#include <atomic>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/options.h>
#include "db/common/file_helper.h"
#include "index/utils/utils.h"

using namespace zvec;
using namespace zvec::test;

namespace {

const std::string kColPath = "test_collection_phase5_5";

class ParallelWritesPhase55Test : public ::testing::Test {
 protected:
  void SetUp() override { FileHelper::RemoveDirectory(kColPath); }
  void TearDown() override { FileHelper::RemoveDirectory(kColPath); }
};

}  // namespace

// --- 4 writer threads inserting disjoint pk ranges into N=4 shards --------
//
// Each thread inserts its own pk range. With per-shard locks, all 4
// threads can run concurrently (each pk routes to one of the 4 shards;
// the ranges are chosen to spread across shards, though not perfectly
// evenly due to hash distribution).
//
// After all threads join, every pk must be fetchable and the total
// doc_count must match the sum of all inserts.
TEST_F(ParallelWritesPhase55Test, ConcurrentInsertsDontLoseData) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = CollectionOptions{false, true, 64 * 1024 * 1024,
                                WalDurability::PER_BATCH, 4};
  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  constexpr int kThreads = 4;
  constexpr int kDocsPerThread = 250;
  std::atomic<int> errors{0};

  auto writer = [&](int thread_id) {
    const int base = thread_id * kDocsPerThread;
    // Insert in small batches to exercise the per-shard lock path
    // multiple times per thread.
    for (int start = 0; start < kDocsPerThread; start += 50) {
      std::vector<Doc> batch;
      for (int i = start; i < std::min(start + 50, kDocsPerThread); ++i) {
        batch.push_back(TestHelper::CreateDoc(base + i, *schema));
      }
      auto r = collection->Insert(batch);
      if (!r.has_value()) {
        errors.fetch_add(1);
        return;
      }
      for (const auto &s : r.value()) {
        if (!s.ok()) errors.fetch_add(1);
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back(writer, t);
  }
  for (auto &th : threads) th.join();

  EXPECT_EQ(errors.load(), 0) << "some inserts failed";

  // Verify every pk is fetchable.
  for (int i = 0; i < kThreads * kDocsPerThread; ++i) {
    auto fetched = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(fetched.has_value());
    auto it = fetched.value().find(TestHelper::MakePK(i));
    ASSERT_NE(it, fetched.value().end());
    EXPECT_NE(it->second, nullptr) << "pk_" << i << " lost after concurrent insert";
  }

  auto stats = collection->Stats().value();
  EXPECT_EQ(stats.doc_count,
            static_cast<uint64_t>(kThreads * kDocsPerThread));
}

// --- Concurrent inserts + concurrent reads don't block or crash -----------
//
// One thread writes; another continuously reads (Fetch). With shared
// write_mtx_, the read thread should never be blocked for the full
// duration of the write thread's lifetime.
TEST_F(ParallelWritesPhase55Test, ConcurrentInsertAndFetchNoDataRace) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = CollectionOptions{false, true, 64 * 1024 * 1024,
                                WalDurability::PER_BATCH, 4};
  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  std::atomic<bool> done{false};
  std::atomic<int> successful_fetches{0};

  // Writer thread: insert 500 docs.
  std::thread writer([&] {
    for (int start = 0; start < 500; start += 50) {
      std::vector<Doc> batch;
      for (int i = start; i < start + 50; ++i) {
        batch.push_back(TestHelper::CreateDoc(i, *schema));
      }
      auto r = collection->Insert(batch);
      ASSERT_TRUE(r.has_value()) << r.error().message();
    }
    done.store(true);
  });

  // Reader thread: keep fetching pk_0 until the writer finishes.
  std::thread reader([&] {
    while (!done.load()) {
      auto fetched = collection->Fetch({TestHelper::MakePK(0)});
      if (fetched.has_value()) {
        successful_fetches.fetch_add(1);
      }
      std::this_thread::yield();
    }
  });

  writer.join();
  reader.join();

  // The reader should have completed at least a few fetches while the
  // writer was running — if not, it was blocked the entire time.
  EXPECT_GT(successful_fetches.load(), 0)
      << "reader thread was blocked for the entire write duration";
}

// --- Concurrent upserts to same pks don't corrupt state -------------------
//
// Two threads upsert overlapping pk ranges. Each upsert should succeed
// (no duplicates, no crashes). The final state should have every pk.
TEST_F(ParallelWritesPhase55Test, ConcurrentUpsertsOverlappingPks) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = CollectionOptions{false, true, 64 * 1024 * 1024,
                                WalDurability::PER_BATCH, 4};
  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  // Seed pk_0..pk_99.
  std::vector<Doc> seed;
  for (int i = 0; i < 100; ++i) seed.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->Insert(seed).has_value());

  std::atomic<int> errors{0};

  // Two threads upsert overlapping pk ranges: [0..149] and [50..199].
  auto upserter = [&](int start, int end) {
    for (int base = start; base < end; base += 50) {
      std::vector<Doc> batch;
      for (int i = base; i < std::min(base + 50, end); ++i) {
        batch.push_back(TestHelper::CreateDoc(i, *schema));
      }
      auto r = collection->Upsert(batch);
      if (!r.has_value()) {
        errors.fetch_add(1);
        return;
      }
      for (const auto &s : r.value()) {
        if (!s.ok()) errors.fetch_add(1);
      }
    }
  };

  std::thread t1(upserter, 0, 150);
  std::thread t2(upserter, 50, 200);
  t1.join();
  t2.join();

  EXPECT_EQ(errors.load(), 0);

  // All pks 0..199 should be fetchable.
  for (int i = 0; i < 200; ++i) {
    auto fetched = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(fetched.has_value());
    auto it = fetched.value().find(TestHelper::MakePK(i));
    ASSERT_NE(it, fetched.value().end());
    EXPECT_NE(it->second, nullptr) << "pk_" << i << " missing after concurrent upsert";
  }
}
