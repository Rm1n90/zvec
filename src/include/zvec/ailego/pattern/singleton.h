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

#include <type_traits>

namespace zvec {
namespace ailego {

/*! Singleton (C++11)
 */
template <typename T>
class Singleton {
 public:
  using ObjectType = typename std::remove_reference<T>::type;

  //! Retrieve instance of object.
  //! The instance is heap-allocated and intentionally never destroyed.
  //! This prevents "mutex lock failed: Invalid argument" crashes during
  //! process shutdown when embedding libraries (Python, Node.js) tear
  //! down their runtimes before C++ static destructors run.  The OS
  //! reclaims all process memory at exit regardless.
  static ObjectType &Instance(void) noexcept(
      std::is_nothrow_constructible<ObjectType>::value) {
    static ObjectType *obj = new ObjectType;
    return *obj;
  }

 protected:
  //! Constructor (Allow inheritance)
  Singleton(void) {}

 private:
  //! Disable them
  Singleton(Singleton const &) = delete;
  Singleton(Singleton &&) = delete;
  Singleton &operator=(Singleton const &) = delete;
  Singleton &operator=(Singleton &&) = delete;
};

}  // namespace ailego
}  // namespace zvec
