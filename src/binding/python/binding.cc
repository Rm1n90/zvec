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

#include "python_collection.h"
#include "python_config.h"
#include "python_doc.h"
#include "python_param.h"
#include "python_schema.h"
#include "python_type.h"
#include <zvec/ailego/buffer/buffer_manager.h>
#include "db/common/global_resource.h"

namespace zvec {
PYBIND11_MODULE(_zvec, m) {
  m.doc() = "Zvec core module";

  ZVecPyTyping::Initialize(m);
  ZVecPyParams::Initialize(m);
  ZVecPySchemas::Initialize(m);
  ZVecPyConfig::Initialize(m);
  ZVecPyDoc::Initialize(m);
  ZVecPyCollection::Initialize(m);

  // Expose a shutdown helper that Python's atexit can call. C++ static
  // singletons (GlobalResource, BufferManager) must be torn down while
  // the OS threading infrastructure is still alive. Registering via
  // Python's atexit (rather than a pybind11 cpp_function) ensures the
  // handler survives module cleanup.
  m.def("_shutdown", []() {
    try {
      GlobalResource::Instance().shutdown();
    } catch (...) {}
    try {
      ailego::BufferManager::Instance().cleanup();
    } catch (...) {}
  });
}
}  // namespace zvec
