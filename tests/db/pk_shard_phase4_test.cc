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
// Tests for Phase 4 of the optimization pipeline:
//   - PkToShard: deterministic, portable, well-distributed hash-mod-N.
//   - Read-side accessor invariants: with N_shards = 1, the query path
//     still sees every doc a writer puts in, because get_all_segments()
//     fans out over all_writing_segments() — the Phase-4 no-op
//     iteration is exercised end-to-end.
//
// Phase 5 will flip N_shards > 1 and most of the correctness guarantees
// this file locks in will carry over unchanged.

#include <string>
#include <unordered_map>
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
const std::string kColPath = "test_collection_phase4";

class PkShardPhase4Test : public ::testing::Test {
 protected:
  void SetUp() override { FileHelper::RemoveDirectory(kColPath); }
  void TearDown() override { FileHelper::RemoveDirectory(kColPath); }
};
}  // namespace

// --- PkToShard: n_shards <= 1 always returns 0 ----------------------------
TEST_F(PkShardPhase4Test, PkToShardDegenerateCases) {
  EXPECT_EQ(PkToShard("anything", 0), 0u);
  EXPECT_EQ(PkToShard("anything", 1), 0u);
  EXPECT_EQ(PkToShard("", 1), 0u);
}

// --- PkToShard: deterministic across calls --------------------------------
TEST_F(PkShardPhase4Test, PkToShardDeterministic) {
  // The function must map a given pk → shard identically across calls.
  // Critical for Phase 5: an upsert must land on the same shard as the
  // original insert.
  for (int i = 0; i < 64; ++i) {
    std::string pk = "pk_" + std::to_string(i);
    for (size_t n : {4u, 8u, 16u, 32u}) {
      size_t a = PkToShard(pk, n);
      size_t b = PkToShard(pk, n);
      EXPECT_EQ(a, b) << "PkToShard not deterministic for pk='" << pk
                      << "', n=" << n;
      EXPECT_LT(a, n);
    }
  }
}

// --- PkToShard: distribution is reasonable --------------------------------
//
// Not a strict statistical test — just a sanity guard that the hash is
// doing something better than "always shard 0". 2000 sequential pks
// into 8 shards should produce every shard with at least some load.
TEST_F(PkShardPhase4Test, PkToShardDistribution) {
  constexpr size_t N = 8;
  constexpr int kSamples = 2000;

  std::unordered_map<size_t, int> counts;
  for (int i = 0; i < kSamples; ++i) {
    auto s = PkToShard("pk_" + std::to_string(i), N);
    counts[s]++;
  }

  EXPECT_EQ(counts.size(), N) << "every shard should see at least one pk";
  // Each shard should get at least ~kSamples/(4*N) entries. With good
  // distribution we'd expect kSamples/N = 250; we accept anything above
  // 62 to stay robust to hash quirks.
  for (size_t s = 0; s < N; ++s) {
    EXPECT_GT(counts[s], kSamples / (4 * N))
        << "shard " << s << " under-loaded (" << counts[s] << ")";
  }
}

// --- Integration: after insert, query sees the writing-segment data -------
//
// This test exists to break if the Phase 4 refactor accidentally stops
// iterating all_writing_segments() in get_all_segments(). With data
// only in the writing segment (no flush), a query must still find it.
TEST_F(PkShardPhase4Test, QueryFindsDocInWritingSegmentPostRefactor) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts,
                                                        0, 0, false);
  ASSERT_NE(collection, nullptr);

  // Put a handful of docs into the writing segment. We deliberately do
  // NOT call Flush or Optimize — data should sit in the writing
  // segment only.
  std::vector<Doc> docs;
  for (int i = 0; i < 20; ++i) docs.push_back(TestHelper::CreateDoc(i, *schema));
  auto r = collection->Insert(docs);
  ASSERT_TRUE(r.has_value()) << r.error().message();

  // Fetch by pk — this exercises get_all_segments() on the Query side.
  for (int i = 0; i < 20; ++i) {
    auto fetched = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(fetched.has_value()) << "pk_" << i;
    EXPECT_EQ(fetched.value().size(), 1u)
        << "pk_" << i << " not visible from writing segment";
  }

  auto stats = collection->Stats().value();
  EXPECT_EQ(stats.doc_count, 20u);
}
