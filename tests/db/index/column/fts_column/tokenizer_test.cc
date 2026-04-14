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
#include "db/index/column/fts_column/tokenizer.h"


using namespace zvec;


TEST(DefaultTokenizerTest, EmptyStringProducesNoTokens) {
  DefaultTokenizer t;
  auto out = t.Tokenize("");
  EXPECT_TRUE(out.empty());
}


TEST(DefaultTokenizerTest, WhitespaceOnlyProducesNoTokens) {
  DefaultTokenizer t;
  auto out = t.Tokenize("   \t\n  ");
  EXPECT_TRUE(out.empty());
}


TEST(DefaultTokenizerTest, SingleWordLowercased) {
  DefaultTokenizer t;
  auto out = t.Tokenize("Hello");
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0], "hello");
}


TEST(DefaultTokenizerTest, MixedCaseWordsLowercased) {
  DefaultTokenizer t;
  auto out = t.Tokenize("The Quick Brown Fox");
  std::vector<std::string> expected{"the", "quick", "brown", "fox"};
  EXPECT_EQ(out, expected);
}


TEST(DefaultTokenizerTest, PunctuationSplitsTokens) {
  DefaultTokenizer t;
  auto out = t.Tokenize("hello, world! it's 2026.");
  std::vector<std::string> expected{"hello", "world", "it", "s", "2026"};
  EXPECT_EQ(out, expected);
}


TEST(DefaultTokenizerTest, RepeatedDelimitersIgnored) {
  DefaultTokenizer t;
  auto out = t.Tokenize("   foo---bar    baz ");
  std::vector<std::string> expected{"foo", "bar", "baz"};
  EXPECT_EQ(out, expected);
}


TEST(DefaultTokenizerTest, StopwordsFiltered) {
  DefaultTokenizer::Options opts;
  opts.stopwords = {"the", "a"};
  DefaultTokenizer t(opts);
  auto out = t.Tokenize("The quick brown fox jumps over a lazy dog");
  std::vector<std::string> expected{"quick", "brown", "fox",
                                    "jumps", "over",  "lazy",
                                    "dog"};
  EXPECT_EQ(out, expected);
}


TEST(DefaultTokenizerTest, MinTokenLengthFilters) {
  DefaultTokenizer::Options opts;
  opts.min_token_length = 3;
  DefaultTokenizer t(opts);
  auto out = t.Tokenize("a ab abc abcd");
  std::vector<std::string> expected{"abc", "abcd"};
  EXPECT_EQ(out, expected);
}


TEST(DefaultTokenizerTest, MaxTokenLengthTruncates) {
  DefaultTokenizer::Options opts;
  opts.max_token_length = 5;
  DefaultTokenizer t(opts);
  // The long "abcdefghij" is truncated to "abcde"; the rest of the
  // run is not re-emitted as a second token.
  auto out = t.Tokenize("abcdefghij next");
  std::vector<std::string> expected{"abcde", "next"};
  EXPECT_EQ(out, expected);
}


TEST(DefaultTokenizerTest, LowercaseDisabledPreservesCase) {
  DefaultTokenizer::Options opts;
  opts.lowercase = false;
  DefaultTokenizer t(opts);
  auto out = t.Tokenize("Hello World");
  std::vector<std::string> expected{"Hello", "World"};
  EXPECT_EQ(out, expected);
}


TEST(DefaultTokenizerTest, DigitsAreTokenCharacters) {
  DefaultTokenizer t;
  auto out = t.Tokenize("version 4.6 build 123");
  std::vector<std::string> expected{"version", "4", "6", "build", "123"};
  EXPECT_EQ(out, expected);
}


TEST(DefaultTokenizerTest, AppendsToExistingVector) {
  DefaultTokenizer t;
  std::vector<std::string> out{"pre"};
  t.Tokenize("hello world", &out);
  std::vector<std::string> expected{"pre", "hello", "world"};
  EXPECT_EQ(out, expected);
}


TEST(DefaultTokenizerTest, Utf8BytesPreservedWithinToken) {
  DefaultTokenizer t;
  // "café" in UTF-8 is 63 61 66 C3 A9. We expect the non-ASCII bytes
  // to be preserved as part of the same token.
  auto out = t.Tokenize("caf\xC3\xA9 latte");
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[1], "latte");
  // The first token starts with "caf" (lowercased) followed by the
  // two-byte UTF-8 sequence for 'é'.
  EXPECT_EQ(out[0].size(), 5u);
  EXPECT_EQ(out[0][0], 'c');
  EXPECT_EQ(out[0][1], 'a');
  EXPECT_EQ(out[0][2], 'f');
  EXPECT_EQ(static_cast<unsigned char>(out[0][3]), 0xC3u);
  EXPECT_EQ(static_cast<unsigned char>(out[0][4]), 0xA9u);
}


TEST(DefaultTokenizerTest, Utf8WordsSeparatedBySpacesTokenizeIndividually) {
  DefaultTokenizer t;
  // Two CJK words separated by a space — each should be one token.
  auto out = t.Tokenize("\xE4\xBD\xA0\xE5\xA5\xBD \xE4\xB8\x96\xE7\x95\x8C");
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], "\xE4\xBD\xA0\xE5\xA5\xBD");
  EXPECT_EQ(out[1], "\xE4\xB8\x96\xE7\x95\x8C");
}


TEST(DefaultTokenizerTest, NameIsStable) {
  DefaultTokenizer t;
  EXPECT_EQ(t.Name(), "default");
}


TEST(FtsTokenizerFactoryTest, DefaultNameReturnsDefaultTokenizer) {
  auto t = CreateFtsTokenizer("default");
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(t->Name(), "default");
}


TEST(FtsTokenizerFactoryTest, EmptyNameReturnsDefaultTokenizer) {
  auto t = CreateFtsTokenizer("");
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(t->Name(), "default");
}


TEST(FtsTokenizerFactoryTest, UnknownNameReturnsNull) {
  auto t = CreateFtsTokenizer("not-a-real-tokenizer");
  EXPECT_EQ(t, nullptr);
}
