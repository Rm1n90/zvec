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
// Tests for Phase 6 of the optimization pipeline:
//   - Background AutoOptimizer triggers automatically when the persisted
//     segment count exceeds the configured threshold.
//   - AutoOptimizer lifecycle (start/stop) is clean — no thread leaks
//     or deadlocks on Close.
//   - Manual Optimize() still works when auto-optimize is enabled.

#include <chrono>
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

const std::string kColPath = "test_collection_phase6";

class AutoOptimizePhase6Test : public ::testing::Test {
 protected:
  void SetUp() override { FileHelper::RemoveDirectory(kColPath); }
  void TearDown() override { FileHelper::RemoveDirectory(kColPath); }
};

}  // namespace

// --- Auto-optimize disabled by default: no thread, no trigger -------------
TEST_F(AutoOptimizePhase6Test, DisabledByDefault) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  CollectionOptions opts{false, true, 64 * 1024 * 1024};

  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  // Insert enough docs to create many segments via Flush — but since
  // auto-optimize is OFF, index_completeness stays < 1 (no Optimize
  // was triggered).
  for (int batch = 0; batch < 5; ++batch) {
    std::vector<Doc> docs;
    for (int i = batch * 200; i < (batch + 1) * 200; ++i) {
      docs.push_back(TestHelper::CreateDoc(i, *schema));
    }
    ASSERT_TRUE(collection->Insert(docs).has_value());
    ASSERT_TRUE(collection->Flush().ok());
  }

  auto stats = collection->Stats().value();
  EXPECT_EQ(stats.doc_count, 1000u);
  // Without auto-optimize, vector indices are NOT built on persisted
  // segments (index_completeness < 1 for vector fields).
  EXPECT_LT(stats.index_completeness["dense_fp32"], 1.0);
}

// --- Auto-optimize triggers when segment count exceeds threshold ----------
//
// We set auto_optimize_max_segments = 3, interval = 1 second, and
// cooldown = 0. After inserting + flushing enough to exceed 3 persisted
// segments, the auto-optimizer should trigger within a few seconds and
// bring index_completeness to 1.
TEST_F(AutoOptimizePhase6Test, TriggersOnSegmentCountThreshold) {
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr,
      std::make_shared<HnswIndexParams>(MetricType::IP),
      /*max_doc_count_per_segment=*/1000);
  CollectionOptions opts;
  opts.read_only_ = false;
  opts.enable_mmap_ = true;
  opts.max_buffer_size_ = 64 * 1024 * 1024;
  opts.auto_optimize_enabled_ = true;
  opts.auto_optimize_interval_seconds_ = 1;
  opts.auto_optimize_max_segments_ = 3;
  opts.auto_optimize_cooldown_seconds_ = 0;

  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  // Create 5 persisted segments by inserting 1000 docs per batch (at
  // max_doc_count_per_segment = 1000, each batch fills the writing
  // segment; the NEXT batch's first doc triggers rotation, persisting
  // the old segment). After 5 batches we have 4 persisted segments +
  // 1 writing segment with 1000 docs.
  for (int batch = 0; batch < 5; ++batch) {
    auto s = TestHelper::CollectionInsertDoc(
        collection, batch * 1000, (batch + 1) * 1000);
    ASSERT_TRUE(s.ok()) << s.message();
    ASSERT_TRUE(collection->Flush().ok());
  }

  // index_completeness should be < 1 immediately (Optimize hasn't
  // fired yet, just scheduled).
  auto stats_before = collection->Stats().value();
  EXPECT_LT(stats_before.index_completeness["dense_fp32"], 1.0);

  // Wait for the auto-optimizer to fire. HNSW build on ~5000 vectors
  // can take several seconds, so use a generous timeout (30 s).
  bool optimized = false;
  for (int tick = 0; tick < 60; ++tick) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto stats = collection->Stats().value();
    if (stats.index_completeness["dense_fp32"] >= 1.0) {
      optimized = true;
      break;
    }
  }
  EXPECT_TRUE(optimized)
      << "auto-optimizer should have triggered within 30 seconds";

  // All docs must still be fetchable.
  auto stats_after = collection->Stats().value();
  EXPECT_EQ(stats_after.doc_count, 5000u);
}

// --- Clean lifecycle: Close with auto-optimize running doesn't hang -------
TEST_F(AutoOptimizePhase6Test, CloseWhileAutoOptimizeRunning) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  CollectionOptions opts;
  opts.read_only_ = false;
  opts.enable_mmap_ = true;
  opts.max_buffer_size_ = 64 * 1024 * 1024;
  opts.auto_optimize_enabled_ = true;
  opts.auto_optimize_interval_seconds_ = 1;
  opts.auto_optimize_max_segments_ = 100;  // high threshold — won't trigger
  opts.auto_optimize_cooldown_seconds_ = 0;

  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  std::vector<Doc> docs;
  for (int i = 0; i < 50; ++i) docs.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->Insert(docs).has_value());

  // Let the auto-optimize thread spin for a moment.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Close should complete without hanging — the thread is joined
  // cleanly inside Close.
  collection.reset();

  // Reopen to verify data is intact.
  auto reopened = Collection::Open(kColPath, CollectionOptions{false, true});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto stats = reopened.value()->Stats().value();
  EXPECT_EQ(stats.doc_count, 50u);
}

// --- Manual Optimize + auto-optimize coexist without deadlock -------------
TEST_F(AutoOptimizePhase6Test, ManualOptimizeCoexistsWithAuto) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  CollectionOptions opts;
  opts.read_only_ = false;
  opts.enable_mmap_ = true;
  opts.max_buffer_size_ = 64 * 1024 * 1024;
  opts.auto_optimize_enabled_ = true;
  opts.auto_optimize_interval_seconds_ = 1;
  opts.auto_optimize_max_segments_ = 100;  // won't auto-trigger
  opts.auto_optimize_cooldown_seconds_ = 0;

  auto collection =
      TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0, false);
  ASSERT_NE(collection, nullptr);

  std::vector<Doc> docs;
  for (int i = 0; i < 100; ++i) docs.push_back(TestHelper::CreateDoc(i, *schema));
  ASSERT_TRUE(collection->Insert(docs).has_value());
  ASSERT_TRUE(collection->Flush().ok());

  // Manual Optimize while auto-optimize thread is alive. Must not
  // deadlock (both serialize via ddl_mtx_).
  auto s = collection->Optimize();
  EXPECT_TRUE(s.ok()) << s.message();

  auto stats = collection->Stats().value();
  EXPECT_EQ(stats.doc_count, 100u);
  EXPECT_EQ(stats.index_completeness["dense_fp32"], 1.0);
}
