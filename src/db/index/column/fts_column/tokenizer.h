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


#pragma once


#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>


namespace zvec {


/**
 * @brief Abstract tokenizer used by the full-text-search indexer.
 *
 * A tokenizer converts a piece of text into a list of normalized tokens.
 * Tokens are the unit over which posting lists are built. The same tokenizer
 * instance is used at index time and at query time, so that both sides agree
 * on the vocabulary.
 *
 * Implementations must be thread-safe: Tokenize() may be called concurrently.
 */
class FtsTokenizer {
 public:
  using Ptr = std::shared_ptr<FtsTokenizer>;

  virtual ~FtsTokenizer() = default;

  /// Tokenize @p text and append the resulting tokens into @p out.
  /// The previous contents of @p out are preserved.
  virtual void Tokenize(std::string_view text,
                        std::vector<std::string> *out) const = 0;

  /// Convenience overload that returns a fresh vector.
  std::vector<std::string> Tokenize(std::string_view text) const {
    std::vector<std::string> out;
    Tokenize(text, &out);
    return out;
  }

  /// Short, stable name for this tokenizer (for serialization / diagnostics).
  virtual const std::string &Name() const = 0;
};


/**
 * @brief Zero-dependency default tokenizer.
 *
 * Behavior:
 *   - Splits on any ASCII byte that is not a letter, digit, or UTF-8
 *     continuation / leading byte.
 *   - Bytes with the high bit set (0x80..0xFF) are preserved as part of the
 *     current token, so UTF-8 multi-byte sequences are never split mid
 *     codepoint.
 *   - ASCII letters are lowercased when @c lowercase is true.
 *   - Tokens shorter than @c min_token_length are dropped.
 *   - Tokens longer than @c max_token_length are truncated to the limit.
 *   - Tokens present in @c stopwords (after normalization) are dropped.
 *
 * This works well for English and other space-separated Latin-script
 * languages. For CJK or other languages without word delimiters, plug in a
 * language-specific tokenizer through the factory.
 */
class DefaultTokenizer : public FtsTokenizer {
 public:
  struct Options {
    bool lowercase = true;
    std::size_t min_token_length = 1;
    std::size_t max_token_length = 128;
    std::unordered_set<std::string> stopwords{};
  };

  DefaultTokenizer() = default;
  explicit DefaultTokenizer(Options options) : options_(std::move(options)) {}

  using FtsTokenizer::Tokenize;  // Expose the single-argument overload.
  void Tokenize(std::string_view text,
                std::vector<std::string> *out) const override;
  const std::string &Name() const override;

  const Options &options() const {
    return options_;
  }

 private:
  Options options_{};
};


/**
 * @brief Create a tokenizer by name.
 *
 * Currently only @c "default" is registered; unknown names return nullptr.
 * Future tokenizers (ICU, CJK segmenters, etc.) can plug in here without
 * changing any caller.
 */
FtsTokenizer::Ptr CreateFtsTokenizer(const std::string &name);


}  // namespace zvec
