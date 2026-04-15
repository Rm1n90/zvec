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
// Tests for Phase 2 of the optimization pipeline:
//   - WalDurability enum and CollectionOptions.wal_durability_
//   - LocalWalFile::append_batch + group-commit fsync
//   - Segment::{Insert,Upsert,Update,Delete}Batch entries
//   - CollectionImpl::write_impl + Delete using the batched path
//
// Phase 1 tests live in optimize_phase1_test.cc; this file is dedicated
// to write-path / WAL behaviour so a regression here points cleanly at
// the Phase 2 surgery.

#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/options.h>
#include <zvec/db/status.h>
#include "db/common/file_helper.h"
#include "index/utils/utils.h"

using namespace zvec;
using namespace zvec::test;

namespace {

const std::string kColPath = "test_collection_phase2";

class WalDurabilityPhase2Test : public ::testing::Test {
 protected:
  void SetUp() override { FileHelper::RemoveDirectory(kColPath); }
  void TearDown() override { FileHelper::RemoveDirectory(kColPath); }
};

CollectionOptions MakeOptions(WalDurability d) {
  CollectionOptions o;
  o.read_only_ = false;
  o.enable_mmap_ = true;
  o.max_buffer_size_ = 64 * 1024 * 1024;
  o.wal_durability_ = d;
  return o;
}

Collection::Ptr MakeCollection(WalDurability d) {
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto opts = MakeOptions(d);
  return TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0,
                                             false);
}

}  // namespace

// --- Sanity: enum values + struct defaults ---------------------------------
TEST_F(WalDurabilityPhase2Test, EnumDefaultsAndStructDefaults) {
  // The default for both CollectionOptions and SegmentOptions is PER_BATCH.
  CollectionOptions co;
  EXPECT_EQ(co.wal_durability_, WalDurability::PER_BATCH);
  SegmentOptions so{false, true, 1024 * 1024};
  EXPECT_EQ(so.wal_durability_, WalDurability::PER_BATCH);

  // Equality operator should consider wal_durability_.
  CollectionOptions a;
  CollectionOptions b;
  b.wal_durability_ = WalDurability::PER_DOC;
  EXPECT_NE(a, b);
}

// --- All three durability modes succeed end-to-end -------------------------
//
// Each mode must successfully insert, query, and round-trip a batch of
// docs. We don't assert on fsync timing — that's environment-dependent —
// only on functional correctness.
TEST_F(WalDurabilityPhase2Test, AllDurabilityModesRoundTrip) {
  for (WalDurability d :
       {WalDurability::NONE, WalDurability::PER_BATCH, WalDurability::PER_DOC}) {
    SCOPED_TRACE("durability=" + std::to_string(static_cast<int>(d)));
    FileHelper::RemoveDirectory(kColPath);

    auto schema = TestHelper::CreateSchemaWithVectorIndex();
    auto opts = MakeOptions(d);
    auto collection =
        TestHelper::CreateCollectionWithDoc(kColPath, *schema, opts, 0, 0,
                                            false);
    ASSERT_NE(collection, nullptr);

    // Insert 100 docs in a single batch.
    std::vector<Doc> docs;
    for (int i = 0; i < 100; ++i) {
      docs.push_back(TestHelper::CreateDoc(i, *schema));
    }
    auto r = collection->Insert(docs);
    ASSERT_TRUE(r.has_value()) << r.error().message();
    ASSERT_EQ(r.value().size(), 100u);
    for (const auto &s : r.value()) {
      EXPECT_TRUE(s.ok()) << s.message();
    }

    // Fetch one back.
    auto fetched =
        collection->Fetch({TestHelper::MakePK(50)});
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched.value().size(), 1u);
  }
}

// --- Batched Insert returns per-doc statuses aligned with input -----------
//
// Mid-batch duplicates must surface as AlreadyExists at the right input
// indices; the remaining inserts in the same batch must still succeed.
TEST_F(WalDurabilityPhase2Test, BatchInsertHandlesDuplicatesPerDoc) {
  auto collection = MakeCollection(WalDurability::PER_BATCH);
  ASSERT_NE(collection, nullptr);
  auto schema = collection->Schema().value();

  // Pre-seed pk_5.
  std::vector<Doc> seed{TestHelper::CreateDoc(5, schema)};
  ASSERT_TRUE(collection->Insert(seed).has_value());

  // Batch of 10 docs; index 5 collides with the existing pk_5.
  std::vector<Doc> batch;
  for (int i = 0; i < 10; ++i) batch.push_back(TestHelper::CreateDoc(i, schema));

  auto r = collection->Insert(batch);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  ASSERT_EQ(r.value().size(), 10u);

  for (size_t i = 0; i < r.value().size(); ++i) {
    if (i == 5) {
      EXPECT_FALSE(r.value()[i].ok())
          << "expected AlreadyExists at input index 5";
      EXPECT_EQ(r.value()[i].code(), StatusCode::ALREADY_EXISTS);
    } else if (i < 5) {
      // Already inserted in seed batch attempt? No — only pk_5 was seeded.
      // So pk_0..pk_4 should be a duplicate of nothing → succeed.
      EXPECT_TRUE(r.value()[i].ok()) << r.value()[i].message();
    } else {
      EXPECT_TRUE(r.value()[i].ok()) << r.value()[i].message();
    }
  }
}

// --- Batched Update routes pre-merge fetch errors to the input index ------
TEST_F(WalDurabilityPhase2Test, BatchUpdateRoutesPreMergeErrors) {
  auto collection = MakeCollection(WalDurability::PER_BATCH);
  ASSERT_NE(collection, nullptr);
  auto schema = collection->Schema().value();

  // Insert pk_0, pk_1, pk_2.
  std::vector<Doc> seed;
  for (int i = 0; i < 3; ++i) seed.push_back(TestHelper::CreateDoc(i, schema));
  ASSERT_TRUE(collection->Insert(seed).has_value());

  // Update batch: pk_0 (exists), pk_99 (doesn't), pk_2 (exists).
  std::vector<Doc> batch;
  batch.push_back(TestHelper::CreateDoc(0, schema));
  batch.push_back(TestHelper::CreateDoc(99, schema));
  batch.push_back(TestHelper::CreateDoc(2, schema));

  auto r = collection->Update(batch);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  ASSERT_EQ(r.value().size(), 3u);

  EXPECT_TRUE(r.value()[0].ok()) << r.value()[0].message();
  EXPECT_FALSE(r.value()[1].ok()) << "expected NotFound at index 1";
  EXPECT_EQ(r.value()[1].code(), StatusCode::NOT_FOUND);
  EXPECT_TRUE(r.value()[2].ok()) << r.value()[2].message();
}

// --- Batched Delete returns per-pk statuses for missing pks ---------------
TEST_F(WalDurabilityPhase2Test, BatchDeleteHandlesMissingPks) {
  auto collection = MakeCollection(WalDurability::PER_BATCH);
  ASSERT_NE(collection, nullptr);
  auto schema = collection->Schema().value();

  std::vector<Doc> seed;
  for (int i = 0; i < 5; ++i) seed.push_back(TestHelper::CreateDoc(i, schema));
  ASSERT_TRUE(collection->Insert(seed).has_value());

  // Delete pk_2, pk_99 (missing), pk_4.
  std::vector<std::string> pks{TestHelper::MakePK(2), TestHelper::MakePK(99),
                               TestHelper::MakePK(4)};
  auto r = collection->Delete(pks);
  ASSERT_TRUE(r.has_value()) << r.error().message();
  ASSERT_EQ(r.value().size(), 3u);

  EXPECT_TRUE(r.value()[0].ok()) << r.value()[0].message();
  EXPECT_FALSE(r.value()[1].ok());
  EXPECT_EQ(r.value()[1].code(), StatusCode::NOT_FOUND);
  EXPECT_TRUE(r.value()[2].ok()) << r.value()[2].message();
}

// --- PER_BATCH durability survives a normal close+reopen ------------------
//
// Closing a collection cleanly flushes any remaining buffered IO. After
// reopening with the same path, all docs must be readable. This is a
// functional smoke test for the WAL/segment durability path under each
// mode (we cannot easily simulate a real crash here, but the existing
// optimize_recovery / write_recovery suites cover that).
TEST_F(WalDurabilityPhase2Test, ReopenSeesAllBatchedWrites) {
  for (WalDurability d :
       {WalDurability::NONE, WalDurability::PER_BATCH, WalDurability::PER_DOC}) {
    SCOPED_TRACE("durability=" + std::to_string(static_cast<int>(d)));
    FileHelper::RemoveDirectory(kColPath);

    auto schema = TestHelper::CreateSchemaWithVectorIndex();
    auto opts = MakeOptions(d);
    {
      auto collection = TestHelper::CreateCollectionWithDoc(
          kColPath, *schema, opts, 0, 0, false);
      ASSERT_NE(collection, nullptr);

      std::vector<Doc> docs;
      for (int i = 0; i < 50; ++i) {
        docs.push_back(TestHelper::CreateDoc(i, *schema));
      }
      auto r = collection->Insert(docs);
      ASSERT_TRUE(r.has_value()) << r.error().message();
    }

    // Reopen with same options; all docs must be readable.
    auto reopened = Collection::Open(kColPath, opts);
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    auto stats = reopened.value()->Stats().value();
    EXPECT_EQ(stats.doc_count, 50u);

    auto fetched = reopened.value()->Fetch({TestHelper::MakePK(25)});
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched.value().size(), 1u);
  }
}
