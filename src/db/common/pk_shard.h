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
#include <cstdint>
#include <string>
#include <zvec/ailego/hash/crc32c.h>

namespace zvec {

// Deterministic mapping from primary key to write shard.
//
// Properties required by the Phase 4/5 shard model:
//   * Stable across process restarts — a given pk always routes to the
//     same shard (given the same n_shards), which upsert / update /
//     delete correctness depends on.
//   * Portable — Crc32c is byte-deterministic and produces the same
//     value on every platform we target.
//   * Fast — a few ns per call; the write path is sensitive to per-doc
//     overhead.
//
// For n_shards <= 1 the result is always 0. This makes the "Phase 4
// N=1" call sites a no-op vs. raw access to the single writing segment,
// while letting call sites route uniformly now so Phase 5 can flip
// n_shards without touching them again.
inline size_t PkToShard(const std::string &pk, size_t n_shards) noexcept {
  if (n_shards <= 1) return 0;
  const uint32_t h = ailego::Crc32c::Hash(
      reinterpret_cast<const void *>(pk.data()),
      static_cast<size_t>(pk.size()), 0);
  return static_cast<size_t>(h) % n_shards;
}

}  // namespace zvec
