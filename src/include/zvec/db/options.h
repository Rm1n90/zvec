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

#include <cstdint>
#include <zvec/db/cancellation.h>

namespace zvec {

const uint32_t DEFAULT_MAX_BUFFER_SIZE = 64 * 1024 * 1024;  // 128M

struct CollectionOptions {
  bool read_only_{false};
  bool enable_mmap_{true};  // ignnored when load collection
  uint32_t max_buffer_size_{
      DEFAULT_MAX_BUFFER_SIZE};  // ignored when read_only=true

  bool operator==(const CollectionOptions &other) const {
    return read_only_ == other.read_only_ &&
           enable_mmap_ == other.enable_mmap_ &&
           max_buffer_size_ == other.max_buffer_size_;
  }

  bool operator!=(const CollectionOptions &other) const {
    return !(*this == other);
  }

  CollectionOptions() = default;

  CollectionOptions(bool read_only, bool enable_mmap,
                    uint32_t max_buffer_size = DEFAULT_MAX_BUFFER_SIZE)
      : read_only_(read_only),
        enable_mmap_(enable_mmap),
        max_buffer_size_(max_buffer_size) {}
};

struct SegmentOptions {
  bool read_only_;
  bool enable_mmap_;
  uint32_t max_buffer_size_{DEFAULT_MAX_BUFFER_SIZE};
};

struct CreateIndexOptions {
  int concurrency_{0};  // default use config.optimize_thread_pool
};

struct OptimizeOptions {
  // Per-task inner parallelism (threads used inside a single compact/index
  // build task). 0 = use GlobalConfig::optimize_thread_count default.
  //
  // This is the same knob that existed before Phase 1; kept first so that
  // existing callers using `OptimizeOptions{N}` continue to compile and mean
  // "N threads per task" unchanged.
  int concurrency_{0};

  // Maximum number of compact/index tasks allowed to run concurrently.
  // 0 = use GlobalConfig::compact_dispatch_thread_count default.
  // A value of 1 restores the legacy sequential dispatch behaviour.
  int parallel_tasks_{0};

  // Soft memory budget, in bytes, for the whole Optimize call. The admission
  // controller will not dispatch a new task if doing so would push the
  // estimated concurrent peak RSS above this value. 0 = unlimited (bounded
  // only by parallel_tasks_).
  uint64_t memory_budget_bytes_{0};

  // Per-input-doc memory estimate used by the admission controller. Each
  // task's peak memory is estimated as
  //     sum(input_segment.doc_count) * per_doc_memory_estimate_bytes_
  // The default is a conservative 512 B/doc; override when profiling says
  // otherwise.
  uint64_t per_doc_memory_estimate_bytes_{512};

  // Optional cooperative cancellation token. When set and signalled, the
  // Optimize call returns Status::Cancelled at the next check point without
  // applying partial results.
  CancelToken::Ptr cancel_token_{nullptr};
};

struct AddColumnOptions {
  int concurrency_{0};
};

struct AlterColumnOptions {
  int concurrency_{0};
};

}  // namespace zvec