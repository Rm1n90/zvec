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
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/schema.h>


using namespace zvec;


namespace {

constexpr const char *kColPath = "./test_collection_fts";
constexpr const char *kColName = "fts_collection";
constexpr const char *kFtsField = "body";


CollectionSchema MakeSchema() {
  CollectionSchema schema(kColName);
  auto fts_params = std::make_shared<FtsIndexParams>("default", 1.2f, 0.75f);
  schema.add_field(std::make_shared<FieldSchema>(
      kFtsField, DataType::STRING, /*nullable=*/true, fts_params));
  // Collections require at least one vector field — add a small dummy.
  auto vec_params = std::make_shared<FlatIndexParams>(MetricType::IP);
  schema.add_field(std::make_shared<FieldSchema>(
      "vec", DataType::VECTOR_FP32, /*dimension=*/2, /*nullable=*/false,
      vec_params));
  return schema;
}


Doc MakeDoc(const std::string &pk, const std::string &body) {
  Doc d;
  d.set_pk(pk);
  d.set<std::string>(kFtsField, body);
  d.set<std::vector<float>>("vec", std::vector<float>{0.0f, 0.0f});
  return d;
}


std::vector<std::string> ResultPks(const DocPtrList &docs) {
  std::vector<std::string> pks;
  for (const auto &d : docs) {
    if (d) {
      pks.push_back(d->pk());
    }
  }
  return pks;
}

}  // namespace


class CollectionFtsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ailego::FileHelper::RemoveDirectory(kColPath);
  }

  void TearDown() override {
    ailego::FileHelper::RemoveDirectory(kColPath);
  }

  Collection::Ptr OpenFreshCollection() {
    CollectionOptions options;
    options.read_only_ = false;
    options.enable_mmap_ = false;
    auto schema = MakeSchema();
    auto result = Collection::CreateAndOpen(kColPath, schema, options);
    if (!result.has_value()) {
      ADD_FAILURE() << "Collection::CreateAndOpen failed: "
                    << result.error().message();
      return nullptr;
    }
    return result.value();
  }
};


TEST_F(CollectionFtsTest, CreateAndOpenWithFtsField) {
  auto col = OpenFreshCollection();
  ASSERT_NE(col, nullptr);
  auto schema = col->Schema();
  ASSERT_TRUE(schema.has_value());
  EXPECT_TRUE(schema.value().has_field(kFtsField));
}


TEST_F(CollectionFtsTest, BasicQueryReturnsMatchingDoc) {
  auto col = OpenFreshCollection();
  ASSERT_NE(col, nullptr);

  std::vector<Doc> docs{
      MakeDoc("a", "the quick brown fox jumps over the lazy dog"),
      MakeDoc("b", "totally unrelated content"),
      MakeDoc("c", "a fox and a hound"),
  };
  auto write = col->Insert(docs);
  ASSERT_TRUE(write.has_value());

  TextQuery q;
  q.field_name_ = kFtsField;
  q.text_ = "fox";
  q.topk_ = 10;

  auto res = col->QueryText(q);
  ASSERT_TRUE(res.has_value());
  auto pks = ResultPks(res.value());
  ASSERT_EQ(pks.size(), 2u);
  EXPECT_TRUE((pks[0] == "a" && pks[1] == "c") ||
              (pks[0] == "c" && pks[1] == "a"));
}


TEST_F(CollectionFtsTest, ScoresAreAttachedAndDescending) {
  auto col = OpenFreshCollection();
  ASSERT_NE(col, nullptr);

  std::vector<Doc> docs{
      MakeDoc("many", "ranking ranking ranking is great"),
      MakeDoc("once", "a single mention of ranking"),
  };
  auto w = col->Insert(docs);
  ASSERT_TRUE(w.has_value());

  TextQuery q;
  q.field_name_ = kFtsField;
  q.text_ = "ranking";
  q.topk_ = 10;

  auto res = col->QueryText(q);
  ASSERT_TRUE(res.has_value());
  auto &out = res.value();
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0]->pk(), "many");
  EXPECT_EQ(out[1]->pk(), "once");
  EXPECT_GT(out[0]->score(), out[1]->score());
}


TEST_F(CollectionFtsTest, AndModeRequiresAllTerms) {
  auto col = OpenFreshCollection();
  ASSERT_NE(col, nullptr);

  std::vector<Doc> docs{
      MakeDoc("both", "the quick brown fox"),
      MakeDoc("only_quick", "be quick about it"),
      MakeDoc("only_fox", "a fox is a fox"),
  };
  auto w = col->Insert(docs);
  ASSERT_TRUE(w.has_value());

  TextQuery q;
  q.field_name_ = kFtsField;
  q.text_ = "quick fox";
  q.topk_ = 10;
  q.op_ = TextQuery::MatchOp::AND;

  auto res = col->QueryText(q);
  ASSERT_TRUE(res.has_value());
  auto pks = ResultPks(res.value());
  ASSERT_EQ(pks.size(), 1u);
  EXPECT_EQ(pks[0], "both");
}


TEST_F(CollectionFtsTest, TopkLimitsResults) {
  auto col = OpenFreshCollection();
  ASSERT_NE(col, nullptr);

  std::vector<Doc> docs;
  for (int i = 0; i < 5; ++i) {
    docs.push_back(MakeDoc("pk_" + std::to_string(i), "common topic content"));
  }
  auto w = col->Insert(docs);
  ASSERT_TRUE(w.has_value());

  TextQuery q;
  q.field_name_ = kFtsField;
  q.text_ = "common";
  q.topk_ = 3;

  auto res = col->QueryText(q);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(res.value().size(), 3u);
}


TEST_F(CollectionFtsTest, EmptyTextReturnsEmpty) {
  auto col = OpenFreshCollection();
  ASSERT_NE(col, nullptr);

  std::vector<Doc> docs{MakeDoc("a", "anything")};
  ASSERT_TRUE(col->Insert(docs).has_value());

  TextQuery q;
  q.field_name_ = kFtsField;
  q.text_ = "   ";
  q.topk_ = 10;

  auto res = col->QueryText(q);
  ASSERT_TRUE(res.has_value());
  EXPECT_TRUE(res.value().empty());
}


TEST_F(CollectionFtsTest, DeletedDocsAreNotReturned) {
  auto col = OpenFreshCollection();
  ASSERT_NE(col, nullptr);

  std::vector<Doc> docs{
      MakeDoc("alive", "alpha bravo"),
      MakeDoc("doomed", "alpha charlie"),
  };
  ASSERT_TRUE(col->Insert(docs).has_value());

  // Delete one doc — it should disappear from FTS results.
  auto del = col->Delete({"doomed"});
  ASSERT_TRUE(del.has_value());

  TextQuery q;
  q.field_name_ = kFtsField;
  q.text_ = "alpha";
  q.topk_ = 10;

  auto res = col->QueryText(q);
  ASSERT_TRUE(res.has_value());
  auto pks = ResultPks(res.value());
  ASSERT_EQ(pks.size(), 1u);
  EXPECT_EQ(pks[0], "alive");
}


TEST_F(CollectionFtsTest, ValidationRejectsUnknownField) {
  auto col = OpenFreshCollection();
  ASSERT_NE(col, nullptr);

  TextQuery q;
  q.field_name_ = "no_such_field";
  q.text_ = "anything";
  q.topk_ = 10;

  auto res = col->QueryText(q);
  EXPECT_FALSE(res.has_value());
}


TEST_F(CollectionFtsTest, EmptyCollectionReturnsEmpty) {
  auto col = OpenFreshCollection();
  ASSERT_NE(col, nullptr);

  TextQuery q;
  q.field_name_ = kFtsField;
  q.text_ = "anything";
  q.topk_ = 10;

  auto res = col->QueryText(q);
  ASSERT_TRUE(res.has_value());
  EXPECT_TRUE(res.value().empty());
}


TEST_F(CollectionFtsTest, OutputFieldsNarrowsReturnedDoc) {
  auto col = OpenFreshCollection();
  ASSERT_NE(col, nullptr);

  std::vector<Doc> docs{MakeDoc("a", "alpha")};
  ASSERT_TRUE(col->Insert(docs).has_value());

  TextQuery q;
  q.field_name_ = kFtsField;
  q.text_ = "alpha";
  q.topk_ = 10;
  q.output_fields_ = std::vector<std::string>{};  // explicitly empty -> drop all

  auto res = col->QueryText(q);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value().size(), 1u);
  EXPECT_EQ(res.value()[0]->pk(), "a");
  EXPECT_TRUE(res.value()[0]->field_names().empty());
}
