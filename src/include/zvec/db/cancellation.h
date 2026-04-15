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

#include <atomic>
#include <memory>

namespace zvec {

// Cooperative cancellation token for long-running operations (e.g. Optimize).
//
// The token is checked at natural boundaries inside the operation; when
// cancelled, the operation aborts at the next check point and returns
// Status::Cancelled. Any work that had already been committed stays committed;
// work in flight is rolled back to the pre-commit state.
//
// The token is shared_ptr-backed so a caller on one thread can cancel an
// operation running on another thread without lifetime concerns.
class CancelToken {
 public:
  using Ptr = std::shared_ptr<CancelToken>;

  CancelToken() = default;

  // Request cancellation. Idempotent.
  void cancel() noexcept {
    cancelled_.store(true, std::memory_order_release);
  }

  // Check whether cancellation has been requested.
  bool is_cancelled() const noexcept {
    return cancelled_.load(std::memory_order_acquire);
  }

  // Convenience factory for callers that don't want to manage the shared_ptr.
  static Ptr Create() {
    return std::make_shared<CancelToken>();
  }

 private:
  std::atomic<bool> cancelled_{false};
};

}  // namespace zvec
