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


#include "tokenizer.h"

#include <cctype>


namespace zvec {


namespace {

// A byte is part of a token when it is:
//   - an ASCII letter or digit, or
//   - a non-ASCII byte (>= 0x80), which keeps UTF-8 codepoints whole.
// Every other byte (whitespace, punctuation, control chars) is a delimiter.
inline bool is_token_byte(unsigned char c) {
  if (c >= 0x80) {
    return true;
  }
  return std::isalnum(c) != 0;
}

inline char ascii_tolower(unsigned char c) {
  if (c >= 'A' && c <= 'Z') {
    return static_cast<char>(c + ('a' - 'A'));
  }
  return static_cast<char>(c);
}

}  // namespace


void DefaultTokenizer::Tokenize(std::string_view text,
                                std::vector<std::string> *out) const {
  if (out == nullptr || text.empty()) {
    return;
  }

  const std::size_t n = text.size();
  std::size_t i = 0;
  std::string buf;
  buf.reserve(16);

  while (i < n) {
    // Skip delimiter bytes.
    while (i < n && !is_token_byte(static_cast<unsigned char>(text[i]))) {
      ++i;
    }
    if (i >= n) {
      break;
    }

    // Consume a run of token bytes.
    buf.clear();
    while (i < n && is_token_byte(static_cast<unsigned char>(text[i]))) {
      const unsigned char c = static_cast<unsigned char>(text[i]);
      if (options_.lowercase && c < 0x80) {
        buf.push_back(ascii_tolower(c));
      } else {
        buf.push_back(static_cast<char>(c));
      }
      ++i;
      if (buf.size() >= options_.max_token_length) {
        // Truncate; keep scanning forward to the end of the run so we do not
        // accidentally emit the rest as a new token.
        while (i < n && is_token_byte(static_cast<unsigned char>(text[i]))) {
          ++i;
        }
        break;
      }
    }

    if (buf.size() < options_.min_token_length) {
      continue;
    }
    if (!options_.stopwords.empty() &&
        options_.stopwords.find(buf) != options_.stopwords.end()) {
      continue;
    }
    out->push_back(buf);
  }
}


const std::string &DefaultTokenizer::Name() const {
  static const std::string kName = "default";
  return kName;
}


FtsTokenizer::Ptr CreateFtsTokenizer(const std::string &name) {
  if (name.empty() || name == "default") {
    return std::make_shared<DefaultTokenizer>();
  }
  return nullptr;
}


}  // namespace zvec
