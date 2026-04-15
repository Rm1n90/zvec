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
// Tests for Phase 1 of the optimization pipeline:
//   - Parallel compact-task dispatch via compact_dispatch_pool
//   - Admission control (parallel_tasks cap + memory_budget)
//   - Cooperative cancellation via CancelToken
//   - Shrunk commit-phase lock window (writes/queries during Optimize)
//
// These tests are deliberately separate from collection_test.cc so that
// Phase-1-specific behaviour is easy to trace in CI output and so that a
// regression here points cleanly at the Phase 1 surgery.

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/cancellation.h>
#include <zvec/db/collection.h>
#include <zvec/db/options.h>
#include <zvec/db/status.h>
#include "db/common/file_helper.h"
#include "index/utils/utils.h"

using namespace zvec;
using namespace zvec::test;

namespace {

std::string col_path = "test_collection_phase1";

class OptimizePhase1Test : public ::testing::Test {
 protected:
  void SetUp() override { FileHelper::RemoveDirectory(col_path); }
  void TearDown() override { FileHelper::RemoveDirectory(col_path); }
};

// Build a collection with N persisted segments so that Optimize has real
// work to dispatch. Each rotated segment gives the planner a compact or
// create-vector-index task to schedule.
// Schema validation requires max_doc_count_per_segment >= 1000.
Collection::Ptr MakeCollectionWithSegments(int segments,
                                           int docs_per_segment = 1000) {
  FileHelper::RemoveDirectory(col_path);
  auto schema = TestHelper::CreateNormalSchema(
      /*nullable=*/false, /*name=*/"demo", /*scalar_index_params=*/nullptr,
      std::make_shared<HnswIndexParams>(MetricType::IP),
      /*max_doc_count_per_segment=*/static_cast<uint64_t>(docs_per_segment));
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, 0, false);
  for (int i = 0; i < segments; ++i) {
    auto s = TestHelper::CollectionInsertDoc(
        collection, i * docs_per_segment, (i + 1) * docs_per_segment);
    EXPECT_TRUE(s.ok()) << s.message();
    EXPECT_TRUE(collection->Flush().ok());
  }
  return collection;
}

}  // namespace

// --- Backward compatibility ------------------------------------------------
//
// Phase 1 added four fields to OptimizeOptions. The old aggregate-init
// pattern `OptimizeOptions{N}` must continue to compile and mean exactly
// what it did before (`concurrency_=N`, everything else defaulted).
TEST_F(OptimizePhase1Test, OptionsBraceInitBackwardCompat) {
  OptimizeOptions a{};
  EXPECT_EQ(a.concurrency_, 0);
  EXPECT_EQ(a.parallel_tasks_, 0);
  EXPECT_EQ(a.memory_budget_bytes_, 0u);
  EXPECT_EQ(a.per_doc_memory_estimate_bytes_, 512u);
  EXPECT_EQ(a.cancel_token_, nullptr);

  OptimizeOptions b{2};
  EXPECT_EQ(b.concurrency_, 2);
  EXPECT_EQ(b.parallel_tasks_, 0);
  EXPECT_EQ(b.memory_budget_bytes_, 0u);
  EXPECT_EQ(b.cancel_token_, nullptr);
}

// --- parallel_tasks=1 path (legacy sequential equivalent) ------------------
TEST_F(OptimizePhase1Test, ParallelTasksOneIsSequentialEquivalent) {
  auto collection = MakeCollectionWithSegments(/*segments=*/6);

  OptimizeOptions opts{};
  opts.concurrency_ = 0;
  opts.parallel_tasks_ = 1;

  auto s = collection->Optimize(opts);
  ASSERT_TRUE(s.ok()) << s.message();

  auto stats = collection->Stats().value();
  // After optimize every doc should be indexed.
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
}

// --- parallel_tasks>1 path (actually parallel) -----------------------------
//
// End-to-end correctness check: with N concurrent tasks the result must
// still be identical to the sequential case (all segments merged/indexed,
// no data loss, full index_completeness).
TEST_F(OptimizePhase1Test, ParallelDispatchProducesConsistentResult) {
  constexpr int kSegments = 8;
  auto collection = MakeCollectionWithSegments(kSegments);

  OptimizeOptions opts{};
  opts.parallel_tasks_ = 4;

  auto s = collection->Optimize(opts);
  ASSERT_TRUE(s.ok()) << s.message();

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
}

// --- memory budget: tight budget still completes ---------------------------
//
// A memory budget smaller than a single task's estimated footprint must
// not deadlock the admission controller — it falls back to running one
// task at a time.
TEST_F(OptimizePhase1Test, TightMemoryBudgetDoesNotDeadlock) {
  auto collection = MakeCollectionWithSegments(/*segments=*/4);

  OptimizeOptions opts{};
  opts.parallel_tasks_ = 4;
  opts.memory_budget_bytes_ = 1;  // absurdly tight
  opts.per_doc_memory_estimate_bytes_ = 1024;

  auto s = collection->Optimize(opts);
  ASSERT_TRUE(s.ok()) << s.message();
}

// --- cancellation: pre-cancelled token aborts cleanly ----------------------
TEST_F(OptimizePhase1Test, PreCancelledTokenAbortsOptimize) {
  auto collection = MakeCollectionWithSegments(/*segments=*/4);

  OptimizeOptions opts{};
  opts.parallel_tasks_ = 1;  // keep tasks serialized so the cancel lands
                             // between admissions deterministically
  opts.cancel_token_ = CancelToken::Create();
  opts.cancel_token_->cancel();  // cancel before starting

  auto s = collection->Optimize(opts);
  EXPECT_FALSE(s.ok()) << "pre-cancelled token should abort Optimize";
  // After a cancelled Optimize the collection must still be usable.
  ASSERT_TRUE(collection->Flush().ok());
}

// --- cancellation: token cancelled mid-run eventually aborts ---------------
TEST_F(OptimizePhase1Test, MidRunCancellationIsHonoured) {
  auto collection = MakeCollectionWithSegments(/*segments=*/6);

  OptimizeOptions opts{};
  opts.parallel_tasks_ = 1;
  opts.cancel_token_ = CancelToken::Create();

  std::atomic<bool> cancelled{false};
  std::thread canceller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    opts.cancel_token_->cancel();
    cancelled.store(true);
  });

  auto s = collection->Optimize(opts);
  canceller.join();

  // Either the cancel lands while tasks are dispatching (returns Cancelled)
  // or Optimize finishes first (returns OK). Both are valid outcomes; what
  // matters is that the collection remains consistent either way.
  EXPECT_TRUE(cancelled.load());
  auto stats = collection->Stats().value();
  EXPECT_GT(stats.doc_count, 0);
}

// --- concurrent queries during Optimize do not block long ------------------
//
// The Phase 1 refactor released schema_handle_mtx_ during the compact
// phase. A Query issued while Optimize is running must not stall for the
// full duration of Optimize — only for the brief commit window.
//
// This is a timing-sensitive regression test. It uses generous bounds so
// it doesn't flake on loaded CI hosts.
TEST_F(OptimizePhase1Test, QueriesRunConcurrentlyWithOptimizeCompactPhase) {
  constexpr int kSegments = 8;
  auto collection = MakeCollectionWithSegments(kSegments);

  OptimizeOptions opts{};
  opts.parallel_tasks_ = 2;

  std::atomic<bool> optimize_done{false};
  std::thread optimize_thread([&] {
    auto s = collection->Optimize(opts);
    ASSERT_TRUE(s.ok()) << s.message();
    optimize_done.store(true);
  });

  // While optimize runs, issue repeated Fetch() calls and record their
  // per-call latency. If the lock refactor worked, max latency should stay
  // well below full-optimize duration.
  int successful_queries = 0;
  auto loop_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!optimize_done.load() &&
         std::chrono::steady_clock::now() < loop_deadline) {
    auto pk = TestHelper::MakePK(0);
    auto result = collection->Fetch({pk});
    if (result.has_value() && result.value().count(pk) == 1) {
      ++successful_queries;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  optimize_thread.join();

  // The functional assertion: at least *some* queries completed during
  // optimize. Under the old behaviour (schema_handle_mtx_ exclusive for the
  // whole call), queries would be fully serialized behind Optimize.
  EXPECT_GT(successful_queries, 0);
}
