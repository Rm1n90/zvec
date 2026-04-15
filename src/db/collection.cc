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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include "db/common/global_resource.h"
#include <ailego/io/file_lock.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/options.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "db/common/pk_shard.h"
#include "db/common/profiler.h"
#include "db/common/typedef.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/index_filter.h"
#include "db/index/common/version_manager.h"
#include "db/index/segment/segment.h"
#include "db/index/segment/segment_helper.h"
#include "db/index/segment/segment_manager.h"
#include "db/sqlengine/sqlengine.h"

namespace zvec {

enum class WriteMode : uint8_t {
  UNDEFINED = 0,
  INSERT,
  UPDATE,
  UPSERT,
};

Collection::~Collection() = default;

class CollectionImpl : public Collection {
  friend class Collection;

 public:
  explicit CollectionImpl(const std::string &path,
                          const CollectionSchema &schema);

  explicit CollectionImpl(const std::string &path);

  ~CollectionImpl() override;

 private:
  Status Open(const CollectionOptions &options);

  Status Close();

 public:
  Status Destroy() override;

  Status Flush() override;

  Result<std::string> Path() const override;

  Result<CollectionStats> Stats() const override;

  Result<CollectionSchema> Schema() const override;

  Result<CollectionOptions> Options() const override;

 public:
  Status CreateIndex(const std::string &column_name,
                     const IndexParams::Ptr &index_params,
                     const CreateIndexOptions &options) override;

  Status DropIndex(const std::string &column_name) override;

  Status Optimize(const OptimizeOptions &options) override;

  Status AddColumn(const FieldSchema::Ptr &column_schema,
                   const std::string &expression,
                   const AddColumnOptions &options) override;

  Status DropColumn(const std::string &column_name) override;

  Status AlterColumn(
      const std::string &column_name, const std::string &rename,
      const FieldSchema::Ptr &new_column_schema = nullptr,
      const AlterColumnOptions &options = AlterColumnOptions()) override;

  Result<WriteResults> Insert(std::vector<Doc> &docs) override;

  Result<WriteResults> Upsert(std::vector<Doc> &docs) override;

  Result<WriteResults> Update(std::vector<Doc> &docs) override;

  Result<WriteResults> Delete(const std::vector<std::string> &pks) override;

  Status DeleteByFilter(const std::string &filter) override;

  Result<DocPtrList> Query(const VectorQuery &query) const override;

  Result<DocPtrList> QueryText(const TextQuery &query) const override;

  Result<GroupResults> GroupByQuery(
      const GroupByVectorQuery &query) const override;

  Result<DocPtrMap> Fetch(const std::vector<std::string> &pks) const override;

 private:
  void prepare_schema();

  Status close_unsafe();

  Status flush_unsafe();

  Status create();

  Status recovery();

  Status create_idmap_and_delete_store();

  Status recover_idmap_and_delete_store();

  Status acquire_file_lock(bool create = false);

  Status init_version_manager();

  Status init_writing_segment();

  bool need_switch_to_new_segment() const;

  Status switch_to_new_segment_for_writing(
      const CollectionSchema::Ptr &schema = nullptr);

  Result<WriteResults> write_impl(std::vector<Doc> &docs, WriteMode mode);

  std::vector<Segment::Ptr> get_all_segments() const;

  std::vector<Segment::Ptr> get_all_persist_segments() const;

  // === Phase 4 — read-side shard abstraction ==============================
  // These accessors let query / DDL / iteration sites treat the
  // writing-segment layer as a logical fan-out over N_shards shards.
  // Phase 4 hardcodes N_shards = 1, so they behave as if the collection
  // has exactly one writing segment — matching the pre-Phase-4
  // behaviour exactly. Phase 5 will swap the body of these accessors
  // to a real per-shard writing_segments_ vector without touching any
  // of the call sites.
  static constexpr size_t kPhase4ShardCount = 1;

  size_t n_shards() const { return kPhase4ShardCount; }

  size_t shard_of_pk(const std::string &pk) const {
    return PkToShard(pk, n_shards());
  }

  // Returns the writing segment that owns `shard`. Phase 4: only shard
  // 0 is valid.
  Segment::Ptr writing_segment_of_shard(size_t shard) const {
    (void)shard;  // assert once Phase 5 lands; for now shard must be 0
    return writing_segment_;
  }

  // Returns the writing segment a given pk routes to. Phase 5 will make
  // this non-degenerate.
  Segment::Ptr writing_segment_for_pk(const std::string &pk) const {
    return writing_segment_of_shard(shard_of_pk(pk));
  }

  // Enumerate every writing segment across every shard. Phase 4 always
  // returns a one-element vector.
  std::vector<Segment::Ptr> all_writing_segments() const {
    return {writing_segment_};
  }

  Segment::Ptr local_segment_by_doc_id(
      uint64_t doc_id, const std::vector<Segment::Ptr> &segments) const;

  SegmentID allocate_segment_id() {
    return segment_id_allocator_.fetch_add(1);
  }

  SegmentID allocate_segment_id_for_tmp_segment() {
    return tmp_segment_id_allocator_.fetch_add(1);
  }

  std::vector<SegmentTask::Ptr> build_compact_task(
      const CollectionSchema::Ptr &schema,
      const std::vector<Segment::Ptr> &segments, int concurrency,
      const IndexFilter::Ptr filter);

  // Dispatch compact/create-index tasks. Parallelism is bounded by
  // `options.parallel_tasks_` (falls back to compact_dispatch_pool size when
  // 0), and by `options.memory_budget_bytes_` via an estimated per-task
  // memory footprint (sum(input.doc_count) * per_doc_memory_estimate_bytes_).
  // Cancellation is checked between admissions; any task failure short-
  // circuits further dispatch and returns the first error (but already-
  // running tasks still complete — their results are discarded by the
  // caller on error).
  Status execute_compact_task(std::vector<SegmentTask::Ptr> &tasks,
                              const OptimizeOptions &options) const;

  std::vector<SegmentTask::Ptr> build_create_vector_index_task(
      const std::vector<Segment::Ptr> &segments, const std::string &column,
      const IndexParams::Ptr &index_params, int concurrency);

  std::vector<SegmentTask::Ptr> build_create_scalar_index_task(
      const std::vector<Segment::Ptr> &segments, const std::string &column,
      const IndexParams::Ptr &index_params, int concurrency);

  std::vector<SegmentTask::Ptr> build_drop_vector_index_task(
      const std::vector<Segment::Ptr> &segments, const std::string &column);

  std::vector<SegmentTask::Ptr> build_drop_scalar_index_task(
      const std::vector<Segment::Ptr> &segments, const std::string &column);

  Status execute_tasks(std::vector<SegmentTask::Ptr> &tasks) const;

 private:
  Status handle_upsert(Doc &doc);

  Status handle_update(Doc &doc);

  Status handle_insert(Doc &doc);

  Status internal_fetch_by_doc(const Doc &doc, Doc::Ptr *doc_out);

 private:
  // Helper functions for add/alter/drop column
  Status validate(const std::string &column, const FieldSchema::Ptr &schema,
                  const std::string &expression, const std::string &rename,
                  ColumnOp op);

 private:
  std::string path_;

  bool destroyed_{false};

  CollectionSchema::Ptr schema_;

  CollectionOptions options_;

  // Lock ordering (always acquire in this order to avoid deadlock):
  //   ddl_mtx_   (exclusive during any DDL, including Optimize)
  //     -> schema_handle_mtx_  (shared for reads / writes, exclusive briefly
  //                             during commit-style schema mutations)
  //       -> write_mtx_         (exclusive for writes, rotation, commit)
  //
  // ddl_mtx_ was introduced in Phase 1 of the optimization pipeline: it
  // serialises DDL-with-DDL (and DDL-with-Optimize) so that Optimize can
  // release schema_handle_mtx_ during its long compact phase without another
  // DDL mutating the schema concurrently. Without this mutex, downgrading
  // schema_handle_mtx_ from exclusive to shared (or releasing it mid-call)
  // would race with a concurrent CreateIndex/DropIndex/etc.
  mutable std::mutex ddl_mtx_;
  mutable std::shared_mutex schema_handle_mtx_;
  mutable std::shared_mutex write_mtx_;

  std::atomic<SegmentID> segment_id_allocator_;
  std::atomic<SegmentID> tmp_segment_id_allocator_;

  // writing segment
  Segment::Ptr writing_segment_;
  // non-writing segments, sort by doc_id range
  SegmentManager::Ptr segment_manager_;

  // latest version: std::vector<SegmentMeta>
  VersionManager::Ptr version_manager_;

  // file lock
  ailego::File lock_file_;

  IDMap::Ptr id_map_;
  DeleteStore::Ptr delete_store_;

  sqlengine::SQLEngine::Ptr sql_engine_;
};

Result<Collection::Ptr> Collection::CreateAndOpen(
    const std::string &path, const CollectionSchema &schema,
    const CollectionOptions &options) {
  auto collection = std::make_shared<CollectionImpl>(path, schema);

  auto s = collection->Open(options);
  CHECK_RETURN_STATUS_EXPECTED(s);

  return collection;
}

Result<Collection::Ptr> Collection::Open(const std::string &path,
                                         const CollectionOptions &options) {
  auto collection = std::make_shared<CollectionImpl>(path);

  auto s = collection->Open(options);
  CHECK_RETURN_STATUS_EXPECTED(s);

  return collection;
}

CollectionImpl::CollectionImpl(const std::string &path,
                               const CollectionSchema &schema)
    : path_(path), schema_(std::make_shared<CollectionSchema>(schema)) {
  prepare_schema();
}

void CollectionImpl::prepare_schema() {
  // set default index params for vector fields
  for (auto &field : schema_->fields()) {
    if (field->is_vector_field()) {
      if (field->index_params() == nullptr) {
        field->set_index_params(DefaultVectorIndexParams);
      }
    }
  }
}

CollectionImpl::CollectionImpl(const std::string &path) : path_(path) {}

CollectionImpl::~CollectionImpl() {
  if (!destroyed_) {
    Close();
  }
}

Status CollectionImpl::Open(const CollectionOptions &options) {
  options_ = options;

  if (schema_ != nullptr && options_.read_only_) {
    return Status::InvalidArgument(
        "Unable to create collection with read-only mode.");
  }

  Status s;
  if (schema_ == nullptr) {
    // recovery from disk
    s = recovery();
  } else {
    // create new collection with existing schema
    s = create();
  }

  auto profiler = std::make_shared<Profiler>();
  sql_engine_ = sqlengine::SQLEngine::create(profiler);

  return s;
}

Status CollectionImpl::Close() {
  // only called in deconstructor
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  return close_unsafe();
}

Status CollectionImpl::close_unsafe() {
  Status result = Status::OK();

  // flush
  if (!options_.read_only_) {
    auto s = flush_unsafe();
    if (!s.ok()) {
      result = s;
    }
  }

  // always release resources regardless of flush outcome
  writing_segment_.reset();
  segment_manager_.reset();
  version_manager_.reset();
  id_map_.reset();
  delete_store_.reset();

  lock_file_.close();

  return result;
}

Status CollectionImpl::Destroy() {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  auto s = close_unsafe();
  CHECK_RETURN_STATUS(s);

  ailego::FileHelper::RemoveDirectory(path_.c_str());

  destroyed_ = true;

  return Status::OK();
}

Status CollectionImpl::Flush() {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard lock(schema_handle_mtx_);
  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  return flush_unsafe();
}

Status CollectionImpl::flush_unsafe() {
  if (!writing_segment_) {
    return Status::InternalError(
        "flush writing segment failed because writing segment is nullptr");
  }
  return writing_segment_->flush();
}

Result<std::string> CollectionImpl::Path() const {
  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  return path_;
}

Result<CollectionStats> CollectionImpl::Stats() const {
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  auto segments = get_all_segments();

  CollectionStats stats;
  auto vector_fields = schema_->vector_fields();
  if (segments.empty()) {
    stats.doc_count = 0;
    for (auto &field : vector_fields) {
      stats.index_completeness[field->name()] =
          1;  // if no doc, completeness is 1
    }
    return stats;
  }

  for (auto &segment : segments) {
    stats.doc_count += segment->doc_count(delete_store_->make_filter());
  }

  for (auto &field : vector_fields) {
    if (stats.doc_count == 0) {
      stats.index_completeness[field->name()] = 1;
      continue;
    }

    uint32_t indexed_doc_count{0};
    for (auto &segment : segments) {
      if (segment->meta()->vector_indexed(field->name())) {
        indexed_doc_count += segment->doc_count(delete_store_->make_filter());
      }
    }
    stats.index_completeness[field->name()] =
        indexed_doc_count * 1.0 / stats.doc_count;
  }

  return stats;
}

Result<CollectionSchema> CollectionImpl::Schema() const {
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  return *schema_;
}

Result<CollectionOptions> CollectionImpl::Options() const {
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  return options_;
}

Status CollectionImpl::CreateIndex(const std::string &column_name,
                                   const IndexParams::Ptr &index_params,
                                   const CreateIndexOptions &options) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  // ddl_mtx_ serialises DDL-with-DDL (including Optimize); the exclusive
  // schema_handle_mtx_ below still blocks queries/writes for the duration
  // of this CreateIndex (unchanged behaviour — shrinking this lock scope is
  // a later phase concern).
  std::lock_guard<std::mutex> ddl_lock(ddl_mtx_);
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  auto new_schema = std::make_shared<CollectionSchema>(*schema_);
  auto s = new_schema->add_index(column_name, index_params);
  CHECK_RETURN_STATUS(s);
  s = new_schema->validate();
  CHECK_RETURN_STATUS(s);

  auto field = schema_->get_field(column_name);
  if (field->index_params() != nullptr &&
      *field->index_params() == *index_params) {
    // equal index params
    return Status::OK();
  }

  // forbidden writing until index is ready
  std::lock_guard write_lock(write_mtx_);

  Version new_version = version_manager_->get_current_version();

  if (writing_segment_->doc_count() > 0) {
    s = writing_segment_->dump();
    CHECK_RETURN_STATUS(s);

    s = segment_manager_->add_segment(writing_segment_);
    CHECK_RETURN_STATUS(s);

    auto seg_options =
        SegmentOptions{false, options_.enable_mmap_, options_.max_buffer_size_,
                       options_.wal_durability_};
    auto new_segment = Segment::CreateAndOpen(
        path_, *new_schema, allocate_segment_id(),
        writing_segment_->meta()->max_doc_id() + 1, id_map_, delete_store_,
        version_manager_, seg_options);
    if (!new_segment) {
      return new_segment.error();
    }

    s = new_version.add_persisted_segment_meta(writing_segment_->meta());
    CHECK_RETURN_STATUS(s);

    writing_segment_ = new_segment.value();
    new_version.set_next_segment_id(segment_id_allocator_.load());

  } else {
    // TODO: allocate new segment id and clear current writing segment at last
    // recreate writing segment
    s = writing_segment_->destroy();
    CHECK_RETURN_STATUS(s);
    auto id = writing_segment_->id();
    auto min_doc_id = writing_segment_->meta()->min_doc_id();
    writing_segment_.reset();
    SegmentOptions seg_options;
    seg_options.enable_mmap_ = options_.enable_mmap_;
    seg_options.max_buffer_size_ = options_.max_buffer_size_;
    seg_options.read_only_ = options_.read_only_;
    seg_options.wal_durability_ = options_.wal_durability_;
    auto writing_segment =
        Segment::CreateAndOpen(path_, *new_schema, id, min_doc_id, id_map_,
                               delete_store_, version_manager_, seg_options);
    if (!writing_segment) {
      return writing_segment.error();
    }
    writing_segment_ = writing_segment.value();
  }
  new_version.reset_writing_segment_meta(writing_segment_->meta());

  // get_all_segment will return writing segment if it has docs
  auto persist_segments = get_all_persist_segments();

  bool is_vector_field = field->is_vector_field();

  std::vector<SegmentTask::Ptr> tasks;
  if (is_vector_field) {
    tasks = build_create_vector_index_task(persist_segments, column_name,
                                           index_params, options.concurrency_);

  } else {
    tasks = build_create_scalar_index_task(persist_segments, column_name,
                                           index_params, options.concurrency_);
  }

  if (tasks.empty()) {
    new_version.set_schema(*new_schema);

    s = version_manager_->apply(new_version);
    CHECK_RETURN_STATUS(s);

    // persist manifest
    s = version_manager_->flush();
    CHECK_RETURN_STATUS(s);

    schema_ = new_schema;
    return Status::OK();
  }

  s = execute_tasks(tasks);
  CHECK_RETURN_STATUS(s);

  new_version.set_schema(*new_schema);

  for (auto &task : tasks) {
    auto task_info = task->task_info();

    if (std::holds_alternative<CreateVectorIndexTask>(task_info)) {
      auto create_index_task = std::get<CreateVectorIndexTask>(task_info);
      s = new_version.update_persisted_segment_meta(
          create_index_task.output_segment_meta_);
    } else if (std::holds_alternative<CreateScalarIndexTask>(task_info)) {
      auto create_index_task = std::get<CreateScalarIndexTask>(task_info);
      s = new_version.update_persisted_segment_meta(
          create_index_task.output_segment_meta_);
    }
    CHECK_RETURN_STATUS(s);
  }

  // 2. update version
  s = version_manager_->apply(new_version);
  CHECK_RETURN_STATUS(s);

  // 3. persist version
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  // 4. remove old segments or block
  for (auto &task : tasks) {
    auto task_info = task->task_info();

    if (std::holds_alternative<CreateVectorIndexTask>(task_info)) {
      auto create_index_task = std::get<CreateVectorIndexTask>(task_info);
      s = create_index_task.input_segment_->reload_vector_index(
          *new_schema, create_index_task.output_segment_meta_,
          create_index_task.output_vector_indexers_,
          create_index_task.output_quant_vector_indexers_);
    } else if (std::holds_alternative<CreateScalarIndexTask>(task_info)) {
      auto create_index_task = std::get<CreateScalarIndexTask>(task_info);
      s = create_index_task.input_segment_->reload_scalar_index(
          *new_schema, create_index_task.output_segment_meta_,
          create_index_task.output_scalar_indexer_);
    }
    CHECK_RETURN_STATUS(s);
  }

  schema_ = new_schema;

  return Status::OK();
}

std::vector<SegmentTask::Ptr> CollectionImpl::build_create_vector_index_task(
    const std::vector<Segment::Ptr> &segments, const std::string &column,
    const IndexParams::Ptr &index_params, int concurrency) {
  std::vector<SegmentTask::Ptr> tasks;
  for (auto &segment : segments) {
    if (!segment->vector_index_ready(column, index_params)) {
      tasks.push_back(SegmentTask::CreateCreateVectorIndexTask(
          CreateVectorIndexTask{segment, column, index_params, concurrency}));
    }
  }
  return tasks;
}

std::vector<SegmentTask::Ptr> CollectionImpl::build_create_scalar_index_task(
    const std::vector<Segment::Ptr> &segments, const std::string &column,
    const IndexParams::Ptr &index_params, int concurrency) {
  std::vector<SegmentTask::Ptr> tasks;
  for (auto &segment : segments) {
    tasks.push_back(SegmentTask::CreateCreateScalarIndexTask(
        CreateScalarIndexTask{segment, {column}, index_params, concurrency}));
  }
  return tasks;
}

Status CollectionImpl::execute_tasks(
    std::vector<SegmentTask::Ptr> &tasks) const {
  Status s;
  for (auto &task : tasks) {
    s = SegmentHelper::Execute(task);
    if (!s.ok()) {
      return s;
    }
  }

  return Status::OK();
}

Status CollectionImpl::DropIndex(const std::string &column_name) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;
  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  std::lock_guard<std::mutex> ddl_lock(ddl_mtx_);
  std::lock_guard lock(schema_handle_mtx_);

  auto new_schema = std::make_shared<CollectionSchema>(*schema_);
  auto s = new_schema->drop_index(column_name);
  CHECK_RETURN_STATUS(s);

  auto field = schema_->get_field(column_name);
  if (field->index_params() == nullptr) {
    return Status::OK();  // return ok if not indexed
  }

  if (field->is_vector_field() &&
      *field->index_params() == DefaultVectorIndexParams) {
    return Status::OK();
  }

  // forbidden writing until index is ready
  std::lock_guard write_lock(write_mtx_);

  Version new_version = version_manager_->get_current_version();

  bool is_vector_field = field->is_vector_field();

  if (writing_segment_->doc_count() > 0) {
    s = writing_segment_->dump();
    CHECK_RETURN_STATUS(s);

    s = segment_manager_->add_segment(writing_segment_);
    CHECK_RETURN_STATUS(s);

    auto new_segment =
        Segment::CreateAndOpen(path_, *new_schema, allocate_segment_id(),
                               writing_segment_->meta()->max_doc_id() + 1,
                               id_map_, delete_store_, version_manager_,
                               SegmentOptions{false, options_.enable_mmap_,
                                              options_.max_buffer_size_,
                                              options_.wal_durability_});
    if (!new_segment) {
      return new_segment.error();
    }

    s = new_version.add_persisted_segment_meta(writing_segment_->meta());
    CHECK_RETURN_STATUS(s);

    writing_segment_ = new_segment.value();
    new_version.set_next_segment_id(segment_id_allocator_.load());

  } else {
    // recreate writing segment
    s = writing_segment_->destroy();
    CHECK_RETURN_STATUS(s);
    auto id = writing_segment_->id();
    auto min_doc_id = writing_segment_->meta()->min_doc_id();
    writing_segment_.reset();
    SegmentOptions seg_options;
    seg_options.enable_mmap_ = options_.enable_mmap_;
    seg_options.max_buffer_size_ = options_.max_buffer_size_;
    seg_options.read_only_ = options_.read_only_;
    seg_options.wal_durability_ = options_.wal_durability_;
    auto writing_segment =
        Segment::CreateAndOpen(path_, *new_schema, id, min_doc_id, id_map_,
                               delete_store_, version_manager_, seg_options);
    if (!writing_segment) {
      return writing_segment.error();
    }

    writing_segment_ = writing_segment.value();
  }
  new_version.reset_writing_segment_meta(writing_segment_->meta());

  auto persist_segments = get_all_persist_segments();

  std::vector<SegmentTask::Ptr> tasks;
  if (is_vector_field) {
    tasks = build_drop_vector_index_task(persist_segments, column_name);
  } else {
    tasks = build_drop_scalar_index_task(persist_segments, column_name);
  }

  if (tasks.empty()) {
    new_version.set_schema(*new_schema);

    s = version_manager_->apply(new_version);
    CHECK_RETURN_STATUS(s);

    // persist manifest
    s = version_manager_->flush();
    CHECK_RETURN_STATUS(s);

    schema_ = new_schema;
    return Status::OK();
  }

  s = execute_tasks(tasks);
  CHECK_RETURN_STATUS(s);

  new_version.set_schema(*new_schema);

  for (auto &task : tasks) {
    auto task_info = task->task_info();

    if (std::holds_alternative<DropVectorIndexTask>(task_info)) {
      auto drop_index_task = std::get<DropVectorIndexTask>(task_info);
      s = new_version.update_persisted_segment_meta(
          drop_index_task.output_segment_meta_);
    } else if (std::holds_alternative<DropScalarIndexTask>(task_info)) {
      auto drop_index_task = std::get<DropScalarIndexTask>(task_info);
      s = new_version.update_persisted_segment_meta(
          drop_index_task.output_segment_meta_);
    }
    CHECK_RETURN_STATUS(s);
  }

  s = version_manager_->apply(new_version);
  CHECK_RETURN_STATUS(s);

  // persist manifest
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  // 4. remove old segments or block
  for (auto &task : tasks) {
    auto task_info = task->task_info();

    if (std::holds_alternative<DropVectorIndexTask>(task_info)) {
      auto drop_index_task = std::get<DropVectorIndexTask>(task_info);
      s = drop_index_task.input_segment_->reload_vector_index(
          *new_schema, drop_index_task.output_segment_meta_,
          drop_index_task.output_vector_indexers_);
    } else if (std::holds_alternative<DropScalarIndexTask>(task_info)) {
      auto drop_index_task = std::get<DropScalarIndexTask>(task_info);
      s = drop_index_task.input_segment_->reload_scalar_index(
          *new_schema, drop_index_task.output_segment_meta_,
          drop_index_task.output_scalar_indexer_);
    }
    CHECK_RETURN_STATUS(s);
  }

  schema_ = new_schema;

  return Status::OK();
}

std::vector<SegmentTask::Ptr> CollectionImpl::build_drop_vector_index_task(
    const std::vector<Segment::Ptr> &segments, const std::string &column) {
  std::vector<SegmentTask::Ptr> tasks;
  for (auto &segment : segments) {
    tasks.emplace_back(SegmentTask::CreateDropVectorIndexTask(
        DropVectorIndexTask{segment, column}));
  }
  return tasks;
}

std::vector<SegmentTask::Ptr> CollectionImpl::build_drop_scalar_index_task(
    const std::vector<Segment::Ptr> &segments, const std::string &column) {
  std::vector<SegmentTask::Ptr> tasks;
  for (auto &segment : segments) {
    tasks.emplace_back(SegmentTask::CreateDropScalarIndexTask(
        DropScalarIndexTask(segment, {column})));
  }
  return tasks;
}

Status CollectionImpl::Optimize(const OptimizeOptions &options) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  // === Phase 0 — DDL serialisation ========================================
  // ddl_mtx_ is held for the entire Optimize. It serialises with other DDL
  // (CreateIndex / DropIndex / AddColumn / AlterColumn / DropColumn) and
  // with concurrent Optimize calls, replacing the previous reliance on
  // holding schema_handle_mtx_ exclusively for the entire call.
  std::lock_guard<std::mutex> ddl_lock(ddl_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  const CancelToken::Ptr cancel = options.cancel_token_;
  auto check_cancel = [&]() -> Status {
    if (cancel && cancel->is_cancelled()) {
      return Status::Cancelled("optimize cancelled");
    }
    return Status::OK();
  };

  // === Phase 1 — quiesce (brief) ==========================================
  // Rotate the writing segment so that subsequent writes land in a fresh
  // segment that is NOT part of the compact input set. Writers are blocked
  // only for the duration of the rotation — tens of ms, not the whole
  // Optimize. Queries continue throughout.
  std::vector<Segment::Ptr> persist_segments;
  {
    std::shared_lock<std::shared_mutex> schema_lock(schema_handle_mtx_);
    std::lock_guard<std::shared_mutex> write_lock(write_mtx_);

    if (writing_segment_->doc_count() != 0) {
      auto s = switch_to_new_segment_for_writing();
      if (!s.ok()) return s;
    }

    persist_segments = get_all_persist_segments();
  }

  if (persist_segments.empty()) return Status::OK();

  CHECK_RETURN_STATUS(check_cancel());

  // === Phase 2 — build tasks (no locks held) ==============================
  // delete_store_ has its own internal synchronisation; clone is safe
  // without holding schema/write locks.
  auto delete_store_clone = delete_store_->clone();
  auto tasks =
      build_compact_task(schema_, persist_segments, options.concurrency_,
                         delete_store_clone->make_filter());
  if (tasks.empty()) return Status::OK();

  // === Phase 3 — execute tasks in parallel (no schema/write locks) ========
  // Concurrent writers and queries run freely during this (typically long)
  // phase. DDL is blocked via ddl_mtx_.
  {
    auto s = execute_compact_task(tasks, options);
    CHECK_RETURN_STATUS(s);
  }

  CHECK_RETURN_STATUS(check_cancel());

  // === Phase 4 — pre-stage outputs (no locks held) ========================
  // Rename tmp segment directories to their final locations and pre-open
  // the resulting Segment objects. This is the dominant commit-time cost
  // (mmap + vector index load) and historically ran under write_mtx_; doing
  // it outside the lock unblocks writers during this window.
  //
  // Crash invariant: on a crash between rename and version.flush() the
  // renamed directory is orphaned but not referenced by the manifest. This
  // matches the pre-existing behaviour; recovery handles it by ignoring
  // directories not present in the manifest.
  struct PreStagedOutput {
    bool is_compact{false};
    bool has_output{false};
    // Compact-task output (filled when is_compact && has_output):
    Segment::Ptr pre_opened_segment;  // ready to add to segment_manager_
    // Input segments to destroy after commit (for CompactTask):
    std::vector<Segment::Ptr> inputs_to_destroy;
  };

  std::vector<PreStagedOutput> pre_staged(tasks.size());
  for (size_t i = 0; i < tasks.size(); ++i) {
    auto &task_info = tasks[i]->task_info();
    if (std::holds_alternative<CompactTask>(task_info)) {
      auto &compact_task = std::get<CompactTask>(task_info);
      pre_staged[i].is_compact = true;
      pre_staged[i].inputs_to_destroy = compact_task.input_segments_;

      if (compact_task.output_segment_meta_) {
        pre_staged[i].has_output = true;

        auto tmp_segment_id = compact_task.output_segment_id_;
        auto tmp_segment_path =
            FileHelper::MakeTempSegmentPath(path_, tmp_segment_id);

        auto new_segment_id = allocate_segment_id();
        auto new_segment_path =
            FileHelper::MakeSegmentPath(path_, new_segment_id);

        if (!FileHelper::MoveDirectory(tmp_segment_path, new_segment_path)) {
          return Status::InternalError("move segment directory failed");
        }

        // The output meta's id now reflects the final segment location.
        compact_task.output_segment_meta_->set_id(new_segment_id);

        auto pre_opened = Segment::Open(
            path_, *schema_, *compact_task.output_segment_meta_, id_map_,
            delete_store_, version_manager_,
            SegmentOptions{true, options_.enable_mmap_});
        if (!pre_opened.has_value()) {
          return pre_opened.error();
        }
        pre_staged[i].pre_opened_segment = std::move(pre_opened.value());
      }
    }
  }

  CHECK_RETURN_STATUS(check_cancel());

  // === Phase 5 — commit (brief exclusive locks) ===========================
  // Now take schema_handle_mtx_ exclusive (blocks queries and writes) +
  // write_mtx_ exclusive. The body is intentionally minimal: apply the
  // merged version, fsync the manifest, swap segments in/out of
  // segment_manager_, and reload indexers. Heavy IO already happened in
  // Phase 4.
  {
    std::lock_guard<std::shared_mutex> schema_lock(schema_handle_mtx_);
    std::lock_guard<std::shared_mutex> write_lock(write_mtx_);

    Version new_version = version_manager_->get_current_version();

    for (size_t i = 0; i < tasks.size(); ++i) {
      auto &task_info = tasks[i]->task_info();

      if (std::holds_alternative<CompactTask>(task_info)) {
        auto &compact_task = std::get<CompactTask>(task_info);

        if (compact_task.output_segment_meta_) {
          auto s = new_version.add_persisted_segment_meta(
              compact_task.output_segment_meta_);
          CHECK_RETURN_STATUS(s);
          new_version.set_next_segment_id(segment_id_allocator_.load());
        }

        for (auto &input_segment : compact_task.input_segments_) {
          auto s =
              new_version.remove_persisted_segment_meta(input_segment->id());
          CHECK_RETURN_STATUS(s);
        }
      } else if (std::holds_alternative<CreateVectorIndexTask>(task_info)) {
        auto &create_index_task = std::get<CreateVectorIndexTask>(task_info);
        auto s = new_version.update_persisted_segment_meta(
            create_index_task.output_segment_meta_);
        CHECK_RETURN_STATUS(s);
      }
    }

    auto s = version_manager_->apply(new_version);
    CHECK_RETURN_STATUS(s);

    s = version_manager_->flush();
    CHECK_RETURN_STATUS(s);

    // Swap segments under the manager. This is a pointer-level operation
    // plus reload_vector_index (which itself takes a per-segment lock).
    for (size_t i = 0; i < tasks.size(); ++i) {
      auto &task_info = tasks[i]->task_info();

      if (std::holds_alternative<CompactTask>(task_info)) {
        if (pre_staged[i].has_output) {
          s = segment_manager_->add_segment(pre_staged[i].pre_opened_segment);
          CHECK_RETURN_STATUS(s);
        }
        for (auto &input_segment : pre_staged[i].inputs_to_destroy) {
          s = segment_manager_->destroy_segment(input_segment->id());
          CHECK_RETURN_STATUS(s);
        }
      } else if (std::holds_alternative<CreateVectorIndexTask>(task_info)) {
        auto &create_index_task = std::get<CreateVectorIndexTask>(task_info);
        s = create_index_task.input_segment_->reload_vector_index(
            *schema_, create_index_task.output_segment_meta_,
            create_index_task.output_vector_indexers_,
            create_index_task.output_quant_vector_indexers_);
        CHECK_RETURN_STATUS(s);
      }
    }
  }

  return Status::OK();
}

std::vector<SegmentTask::Ptr> CollectionImpl::build_compact_task(
    const CollectionSchema::Ptr &schema,
    const std::vector<Segment::Ptr> &segments, int concurrency,
    const IndexFilter::Ptr filter) {
  std::vector<SegmentTask::Ptr> tasks;
  if (segments.empty()) return tasks;

  bool rebuild = false;
  size_t current_doc_count = 0;
  size_t current_actual_doc_count = 0;
  for (auto &segment : segments) {
    current_doc_count += segment->doc_count();
    current_actual_doc_count += segment->doc_count(filter);
  }
  if (current_actual_doc_count <
      current_doc_count * (1 - COMPACT_DELETE_RATIO_THRESHOLD)) {
    // if delete ratio is large enough, rebuild
    rebuild = true;
  }

  auto max_doc_count_per_segment = schema->max_doc_count_per_segment();

  std::vector<Segment::Ptr> current_group;
  current_doc_count = 0;
  current_actual_doc_count = 0;

  for (const auto &seg : segments) {
    uint64_t doc_count = seg->doc_count();
    uint64_t actual_doc_count = seg->doc_count(filter);

    if (!current_group.empty()) {
      SegmentTask::Ptr task;
      bool skip_task{false};
      if (rebuild) {
        if (current_actual_doc_count + actual_doc_count >
            max_doc_count_per_segment) {
          // only create SegmentCompactTask when rebuild=true
          task = SegmentTask::CreateCompactTask(
              CompactTask{path_, schema, current_group,
                          allocate_segment_id_for_tmp_segment(), filter,
                          !options_.enable_mmap_, concurrency});
        }
      } else {
        if (current_doc_count + doc_count > max_doc_count_per_segment) {
          // check current_group size
          if (current_group.size() == 1) {
            task =
                SegmentTask::CreateCreateVectorIndexTask(CreateVectorIndexTask{
                    current_group[0], "", nullptr, concurrency});
            skip_task = current_group[0]->all_vector_index_ready();
          } else {
            task = SegmentTask::CreateCompactTask(
                CompactTask{path_, schema, current_group,
                            allocate_segment_id_for_tmp_segment(), nullptr,
                            !options_.enable_mmap_, concurrency});
          }
        }
      }

      if (task) {
        current_group.clear();
        current_doc_count = 0;
        current_actual_doc_count = 0;
        if (!skip_task) {
          tasks.push_back(task);
        }
      }
    }

    current_group.push_back(seg);
    current_doc_count += doc_count;
    current_actual_doc_count += actual_doc_count;
  }

  if (current_group.size() > 0) {
    SegmentTask::Ptr task;
    if (current_group.size() == 1 && !rebuild) {
      task = SegmentTask::CreateCreateVectorIndexTask(
          CreateVectorIndexTask{current_group[0], "", nullptr, concurrency});
    } else {
      task = SegmentTask::CreateCompactTask(CompactTask{
          path_, schema, current_group, allocate_segment_id_for_tmp_segment(),
          rebuild ? filter : nullptr, !options_.enable_mmap_, concurrency});
    }
    tasks.push_back(task);
  }

  return tasks;
}

namespace {

// Estimate peak working-set memory for a single compact/index task. The
// estimate is intentionally rough (we don't have byte-level accounting on
// segments yet) but is stable across tasks so the admission controller can
// compare them fairly. Used only for soft throttling; it never rejects a
// task forever — a task larger than the budget runs alone once all other
// admitted tasks have drained.
uint64_t EstimateTaskMemoryBytes(const SegmentTask::Ptr &task,
                                 uint64_t per_doc_bytes) {
  if (!task) return 0;
  uint64_t total_docs = 0;
  auto &info = task->task_info();
  if (std::holds_alternative<CompactTask>(info)) {
    for (auto &seg : std::get<CompactTask>(info).input_segments_) {
      if (seg) total_docs += seg->doc_count();
    }
  } else if (std::holds_alternative<CreateVectorIndexTask>(info)) {
    auto &t = std::get<CreateVectorIndexTask>(info);
    if (t.input_segment_) total_docs += t.input_segment_->doc_count();
  } else if (std::holds_alternative<CreateScalarIndexTask>(info)) {
    auto &t = std::get<CreateScalarIndexTask>(info);
    if (t.input_segment_) total_docs += t.input_segment_->doc_count();
  } else if (std::holds_alternative<DropVectorIndexTask>(info)) {
    auto &t = std::get<DropVectorIndexTask>(info);
    if (t.input_segment_) total_docs += t.input_segment_->doc_count();
  } else if (std::holds_alternative<DropScalarIndexTask>(info)) {
    auto &t = std::get<DropScalarIndexTask>(info);
    if (t.input_segment_) total_docs += t.input_segment_->doc_count();
  }
  return total_docs * per_doc_bytes;
}

}  // namespace

Status CollectionImpl::execute_compact_task(
    std::vector<SegmentTask::Ptr> &tasks,
    const OptimizeOptions &options) const {
  if (tasks.empty()) return Status::OK();

  auto *dispatch_pool = GlobalResource::Instance().compact_dispatch_pool();

  const uint32_t pool_size =
      static_cast<uint32_t>(std::max<size_t>(1, dispatch_pool->count()));
  uint32_t max_parallel = options.parallel_tasks_ > 0
                              ? static_cast<uint32_t>(options.parallel_tasks_)
                              : pool_size;
  max_parallel = std::min<uint32_t>(
      max_parallel, static_cast<uint32_t>(tasks.size()));
  max_parallel = std::max<uint32_t>(max_parallel, 1u);

  const uint64_t memory_budget = options.memory_budget_bytes_;
  const uint64_t per_doc_bytes =
      options.per_doc_memory_estimate_bytes_ > 0
          ? options.per_doc_memory_estimate_bytes_
          : 512ULL;
  const CancelToken::Ptr cancel = options.cancel_token_;

  // Pre-compute per-task memory estimates once so the admission loop is
  // allocation-free under the mutex.
  std::vector<uint64_t> task_bytes;
  task_bytes.reserve(tasks.size());
  for (auto &t : tasks) {
    task_bytes.push_back(EstimateTaskMemoryBytes(t, per_doc_bytes));
  }

  std::mutex state_mtx;
  std::condition_variable state_cv;
  uint32_t active_count = 0;
  uint64_t active_bytes = 0;
  Status first_error;  // OK unless a task (or cancel) wins the race to set it
  bool aborted = false;

  auto record_result = [&](Status &&s) {
    std::lock_guard<std::mutex> lk(state_mtx);
    if (!s.ok() && first_error.ok()) {
      first_error = std::move(s);
      aborted = true;
    }
  };

  for (size_t i = 0; i < tasks.size(); ++i) {
    if (cancel && cancel->is_cancelled()) {
      std::lock_guard<std::mutex> lk(state_mtx);
      if (first_error.ok()) {
        first_error = Status::Cancelled("optimize cancelled");
      }
      aborted = true;
      break;
    }

    const uint64_t need_bytes = task_bytes[i];

    {
      std::unique_lock<std::mutex> lk(state_mtx);
      state_cv.wait(lk, [&] {
        if (aborted) return true;
        if (active_count >= max_parallel) return false;
        // Memory admission: skip when budget is disabled, the task fits,
        // or nothing else is running (prevents deadlock on an oversized
        // single task vs. a tight budget).
        if (memory_budget == 0) return true;
        if (active_count == 0) return true;
        return active_bytes + need_bytes <= memory_budget;
      });
      if (aborted) break;
      ++active_count;
      active_bytes += need_bytes;
    }

    SegmentTask::Ptr *task_slot = &tasks[i];

    try {
      dispatch_pool->execute(
          [task_slot, need_bytes, &state_mtx, &state_cv, &active_count,
           &active_bytes, &record_result]() {
            Status s;
            try {
              s = SegmentHelper::Execute(*task_slot);
            } catch (const std::exception &e) {
              s = Status::InternalError("compact task threw: ", e.what());
            } catch (...) {
              s = Status::InternalError("compact task threw unknown exception");
            }
            record_result(std::move(s));
            {
              std::lock_guard<std::mutex> lk(state_mtx);
              --active_count;
              active_bytes -= need_bytes;
            }
            state_cv.notify_all();
          });
    } catch (...) {
      // Enqueue failed (e.g. std::bad_alloc). Unwind the admission counters
      // we just incremented and propagate as an internal error so the
      // outer loop terminates cleanly.
      std::lock_guard<std::mutex> lk(state_mtx);
      --active_count;
      active_bytes -= need_bytes;
      if (first_error.ok()) {
        first_error =
            Status::InternalError("compact_dispatch_pool enqueue failed");
      }
      aborted = true;
      state_cv.notify_all();
      break;
    }
  }

  // Wait for all dispatched tasks to drain (even on abort — we must not
  // return while closures still reference our stack frame).
  {
    std::unique_lock<std::mutex> lk(state_mtx);
    state_cv.wait(lk, [&] { return active_count == 0; });
  }

  if (!first_error.ok()) return first_error;
  if (cancel && cancel->is_cancelled()) {
    return Status::Cancelled("optimize cancelled");
  }
  return Status::OK();
}

Status CollectionImpl::validate(const std::string &column,
                                const FieldSchema::Ptr &schema,
                                const std::string &expression,
                                const std::string &rename, ColumnOp op) {
  auto check_data_type = [&](const FieldSchema *field) -> Status {
    if (field->data_type() < DataType::INT32 ||
        field->data_type() > DataType::DOUBLE) {
      return Status::InvalidArgument(
          "Only support basic numeric data type [int32, int64, uint32, uint64, "
          "float, double]: ",
          field->to_string());
    }
    return Status::OK();
  };

  switch (op) {
    case ColumnOp::ADD: {
      if (schema == nullptr) {
        return Status::InvalidArgument("Column schema is null");
      }

      if (schema->name().empty()) {
        return Status::InvalidArgument("Column name is empty");
      }
      if (schema_->has_field(schema->name())) {
        return Status::InvalidArgument("column already exists: ",
                                       schema->name());
      }

      auto s = schema->validate();
      CHECK_RETURN_STATUS(s);

      s = check_data_type(schema.get());
      CHECK_RETURN_STATUS(s);

      if (expression.empty() && !schema->nullable()) {
        return Status::InvalidArgument(
            "Add column is not supported for non-nullable column: ",
            schema->name());
      }

      break;
    }
    case ColumnOp::ALTER: {
      if (column.empty()) {
        return Status::InvalidArgument("column name is empty");
      }

      if (!schema_->has_field(column)) {
        return Status::InvalidArgument("column ", column, " not found");
      }

      if (!rename.empty() && schema) {
        return Status::InvalidArgument(
            "cannot specify both rename and new column schema");
      }

      auto *old_field_schema = schema_->get_field(column);
      auto s = check_data_type(old_field_schema);
      CHECK_RETURN_STATUS(s);

      if (!rename.empty()) {
        // rename case
        if (schema_->has_field(rename)) {
          return Status::InvalidArgument("new column name ", rename,
                                         " already exists");
        }
      } else {
        // schema change case
        if (!schema) {
          return Status::InvalidArgument("New column schema is null");
        }

        s = schema->validate();
        CHECK_RETURN_STATUS(s);

        if (schema->name().empty()) {
          return Status::InvalidArgument("new column schema name is empty");
        }

        if (!schema->nullable() && old_field_schema->nullable()) {
          return Status::InvalidArgument(
              "new column schema is not nullable, but old column schema is "
              "nullable");
        }

        if (*old_field_schema == *schema) {
          // equal schema
          return Status::OK();
        }

        s = check_data_type(schema.get());
        CHECK_RETURN_STATUS(s);
      }

      break;
    }
    case ColumnOp::DROP: {
      if (!schema_->has_field(column)) {
        return Status::InvalidArgument("Column not exists: ", column);
      }

      auto *old_field_schema = schema_->get_field(column);
      auto s = check_data_type(old_field_schema);
      CHECK_RETURN_STATUS(s);
      break;
    }
    default:
      break;
  }

  return Status::OK();
}

Status CollectionImpl::AddColumn(const FieldSchema::Ptr &column_schema,
                                 const std::string &expression,
                                 const AddColumnOptions &options) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard<std::mutex> ddl_lock(ddl_mtx_);
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  // validate
  auto s = validate("", column_schema, expression, "", ColumnOp::ADD);
  CHECK_RETURN_STATUS(s);

  // forbidden writing until index is ready
  std::lock_guard write_lock(write_mtx_);

  auto new_schema = std::make_shared<CollectionSchema>(*schema_);
  s = new_schema->add_field(column_schema);
  CHECK_RETURN_STATUS(s);

  if (writing_segment_->doc_count() > 0) {
    s = switch_to_new_segment_for_writing();
    CHECK_RETURN_STATUS(s);
  }

  Version new_version = version_manager_->get_current_version();

  // add column on segment manager
  s = segment_manager_->add_column(column_schema, expression,
                                   options.concurrency_);
  CHECK_RETURN_STATUS(s);

  // reset writing segment with new schema
  auto id = writing_segment_->id();
  auto min_doc_id = writing_segment_->meta()->min_doc_id();

  s = writing_segment_->destroy();
  CHECK_RETURN_STATUS(s);
  writing_segment_.reset();

  SegmentOptions seg_options;
  seg_options.enable_mmap_ = options_.enable_mmap_;
  seg_options.max_buffer_size_ = options_.max_buffer_size_;
  seg_options.read_only_ = options_.read_only_;
  seg_options.wal_durability_ = options_.wal_durability_;
  auto writing_segment =
      Segment::CreateAndOpen(path_, *new_schema, id, min_doc_id, id_map_,
                             delete_store_, version_manager_, seg_options);
  if (!writing_segment) {
    return writing_segment.error();
  }
  writing_segment_ = writing_segment.value();

  // update new version
  new_version.set_schema(*new_schema);
  new_version.reset_writing_segment_meta(writing_segment_->meta());

  auto new_segment_metas = segment_manager_->get_segments_meta();
  for (auto meta : new_segment_metas) {
    s = new_version.update_persisted_segment_meta(meta);
    CHECK_RETURN_STATUS(s);
  }

  s = version_manager_->apply(new_version);
  CHECK_RETURN_STATUS(s);

  // persist manifest
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  schema_ = new_schema;

  return Status::OK();
}

Status CollectionImpl::DropColumn(const std::string &column_name) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard<std::mutex> ddl_lock(ddl_mtx_);
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  // validate
  auto s = validate(column_name, nullptr, "", "", ColumnOp::DROP);
  CHECK_RETURN_STATUS(s);

  // forbidden writing until index is ready
  std::lock_guard write_lock(write_mtx_);

  auto new_schema = std::make_shared<CollectionSchema>(*schema_);
  s = new_schema->drop_field(column_name);
  CHECK_RETURN_STATUS(s);

  if (writing_segment_->doc_count() > 0) {
    s = switch_to_new_segment_for_writing();
    CHECK_RETURN_STATUS(s);
  }

  Version new_version = version_manager_->get_current_version();

  // drop column on segment manager
  s = segment_manager_->drop_column(column_name);
  CHECK_RETURN_STATUS(s);

  // reset writing segment with new schema
  auto id = writing_segment_->id();
  auto min_doc_id = writing_segment_->meta()->min_doc_id();

  s = writing_segment_->destroy();
  CHECK_RETURN_STATUS(s);
  writing_segment_.reset();

  SegmentOptions seg_options;
  seg_options.enable_mmap_ = options_.enable_mmap_;
  seg_options.max_buffer_size_ = options_.max_buffer_size_;
  seg_options.read_only_ = options_.read_only_;
  seg_options.wal_durability_ = options_.wal_durability_;
  auto writing_segment =
      Segment::CreateAndOpen(path_, *new_schema, id, min_doc_id, id_map_,
                             delete_store_, version_manager_, seg_options);
  if (!writing_segment) {
    return writing_segment.error();
  }
  writing_segment_ = writing_segment.value();

  // update new version
  new_version.set_schema(*new_schema);
  new_version.reset_writing_segment_meta(writing_segment_->meta());

  auto new_segment_metas = segment_manager_->get_segments_meta();
  for (auto meta : new_segment_metas) {
    s = new_version.update_persisted_segment_meta(meta);
    CHECK_RETURN_STATUS(s);
  }

  s = version_manager_->apply(new_version);
  CHECK_RETURN_STATUS(s);

  // persist manifest
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  schema_ = new_schema;

  return Status::OK();
}

Status CollectionImpl::AlterColumn(const std::string &column_name,
                                   const std::string &rename,
                                   const FieldSchema::Ptr &new_column_schema,
                                   const AlterColumnOptions &options) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard<std::mutex> ddl_lock(ddl_mtx_);
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  // validate
  auto s =
      validate(column_name, new_column_schema, "", rename, ColumnOp::ALTER);
  CHECK_RETURN_STATUS(s);

  // forbidden writing until index is ready
  std::lock_guard write_lock(write_mtx_);

  std::shared_ptr<FieldSchema> new_field_schema{nullptr};
  if (!rename.empty()) {
    new_field_schema =
        std::make_shared<FieldSchema>(*schema_->get_field(column_name));
    new_field_schema->set_name(rename);
  } else {
    new_field_schema = std::make_shared<FieldSchema>(*new_column_schema);
  }

  auto new_schema = std::make_shared<CollectionSchema>(*schema_);
  s = new_schema->alter_field(column_name, new_field_schema);
  CHECK_RETURN_STATUS(s);

  if (writing_segment_->doc_count() > 0) {
    s = switch_to_new_segment_for_writing();
    CHECK_RETURN_STATUS(s);
  }

  Version new_version = version_manager_->get_current_version();

  // alter column on segment manager
  s = segment_manager_->alter_column(column_name, new_field_schema,
                                     options.concurrency_);
  CHECK_RETURN_STATUS(s);

  // reset writing segment with new schema
  auto id = writing_segment_->id();
  auto min_doc_id = writing_segment_->meta()->min_doc_id();

  s = writing_segment_->destroy();
  CHECK_RETURN_STATUS(s);
  writing_segment_.reset();

  SegmentOptions seg_options;
  seg_options.enable_mmap_ = options_.enable_mmap_;
  seg_options.max_buffer_size_ = options_.max_buffer_size_;
  seg_options.read_only_ = options_.read_only_;
  seg_options.wal_durability_ = options_.wal_durability_;
  auto writing_segment =
      Segment::CreateAndOpen(path_, *new_schema, id, min_doc_id, id_map_,
                             delete_store_, version_manager_, seg_options);
  if (!writing_segment) {
    return writing_segment.error();
  }
  writing_segment_ = writing_segment.value();

  // update new version
  new_version.set_schema(*new_schema);
  new_version.reset_writing_segment_meta(writing_segment_->meta());

  auto new_segment_metas = segment_manager_->get_segments_meta();
  for (auto meta : new_segment_metas) {
    s = new_version.update_persisted_segment_meta(meta);
    CHECK_RETURN_STATUS(s);
  }

  s = version_manager_->apply(new_version);
  CHECK_RETURN_STATUS(s);

  // persist manifest
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  schema_ = new_schema;

  return Status::OK();
}

Result<WriteResults> CollectionImpl::Insert(std::vector<Doc> &docs) {
  return write_impl(docs, WriteMode::INSERT);
}

Result<WriteResults> CollectionImpl::Update(std::vector<Doc> &docs) {
  return write_impl(docs, WriteMode::UPDATE);
}

Result<WriteResults> CollectionImpl::Upsert(std::vector<Doc> &docs) {
  return write_impl(docs, WriteMode::UPSERT);
}

Status CollectionImpl::internal_fetch_by_doc(const Doc &doc,
                                             Doc::Ptr *doc_out) {
  auto segments = get_all_segments();
  uint64_t doc_id;
  bool has = id_map_->has(doc.pk(), &doc_id);
  if (!has) {
    return Status::NotFound("Document not found");
  }
  if (delete_store_->is_deleted(doc_id)) {
    return Status::NotFound("Document already deleted");
  }

  auto segment = local_segment_by_doc_id(doc_id, segments);
  if (!segment) {
    LOG_WARN("doc_id: %zu segment not found", (size_t)doc_id);
    return Status::InternalError("Segment not found");
  }

  auto old_doc = segment->Fetch(doc_id);
  if (!old_doc) {
    LOG_WARN("doc_id: %zu fetch doc failed", (size_t)doc_id);
    return Status::InternalError("Fetch doc failed");
  }
  *doc_out = old_doc;
  return Status::OK();
}

Status CollectionImpl::handle_upsert(Doc &doc) {
  return writing_segment_->Upsert(doc);
}

Status CollectionImpl::handle_update(Doc &doc) {
  Doc::Ptr old_doc{nullptr};
  auto s = internal_fetch_by_doc(doc, &old_doc);
  CHECK_RETURN_STATUS(s);

  old_doc->merge(doc);
  return writing_segment_->Update(*old_doc);
}

Status CollectionImpl::handle_insert(Doc &doc) {
  return writing_segment_->Insert(doc);
}

Result<WriteResults> CollectionImpl::write_impl(std::vector<Doc> &docs,
                                                WriteMode mode) {
  CHECK_READONLY_RETURN_STATUS_EXPECTED();

  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  for (auto &&doc : docs) {
    auto validate = doc.validate(schema_, mode == WriteMode::UPDATE);
    CHECK_RETURN_STATUS_EXPECTED(validate);
  }

  // TODO: The granularity of the write_lock is too coarse.
  std::lock_guard write_lock(write_mtx_);

  if (docs.size() > kMaxWriteBatchSize) {
    return tl::make_unexpected(Status::InvalidArgument(
        "Too many docs: ", docs.size(),
        " exceeds max write batch size of ", kMaxWriteBatchSize));
  }

  // Output vector pre-sized so pre-merge errors and batch results land at
  // their input indices. Statuses default-construct to OK and are
  // overwritten below.
  WriteResults results(docs.size());

  // Build the working batch:
  //   INSERT/UPSERT — docs[] is the working batch unchanged.
  //   UPDATE        — fetch the existing doc per pk and merge in the
  //                   caller-supplied fields. Per-doc fetch errors land
  //                   in results[] at the input index; only successfully
  //                   merged docs go into the batch dispatch below.
  std::vector<Doc> merged_docs;
  std::vector<Doc> *working_docs;
  std::vector<size_t> working_to_input_idx;

  if (mode == WriteMode::UPDATE) {
    merged_docs.reserve(docs.size());
    working_to_input_idx.reserve(docs.size());
    for (size_t i = 0; i < docs.size(); ++i) {
      Doc::Ptr old_doc{nullptr};
      auto fetch_status = internal_fetch_by_doc(docs[i], &old_doc);
      if (!fetch_status.ok()) {
        results[i] = fetch_status;
        continue;
      }
      old_doc->merge(docs[i]);
      merged_docs.push_back(*old_doc);
      working_to_input_idx.push_back(i);
    }
    working_docs = &merged_docs;
  } else {
    working_to_input_idx.resize(docs.size());
    std::iota(working_to_input_idx.begin(), working_to_input_idx.end(), 0);
    working_docs = &docs;
  }

  if (working_docs->empty()) return results;

  // Rotate the writing segment if it's already at capacity. We do not
  // chunk the batch across segments — the segment may temporarily exceed
  // schema_->max_doc_count_per_segment(); the next write call performs
  // the rotation. The legacy per-doc loop only rotated *between* docs
  // for the same reason, so this preserves the existing
  // grow-then-rotate behaviour for batches that overflow remaining
  // capacity.
  if (need_switch_to_new_segment()) {
    auto s = switch_to_new_segment_for_writing();
    CHECK_RETURN_STATUS_EXPECTED(s);
  }

  Result<WriteResults> batch_results;
  switch (mode) {
    case WriteMode::INSERT:
      batch_results = writing_segment_->InsertBatch(*working_docs);
      break;
    case WriteMode::UPSERT:
      batch_results = writing_segment_->UpsertBatch(*working_docs);
      break;
    case WriteMode::UPDATE:
      batch_results = writing_segment_->UpdateBatch(*working_docs);
      break;
    default:
      return tl::make_unexpected(
          Status::InvalidArgument("Invalid write mode"));
  }
  if (!batch_results.has_value()) {
    return batch_results;
  }

  for (size_t bi = 0; bi < batch_results.value().size(); ++bi) {
    results[working_to_input_idx[bi]] = std::move(batch_results.value()[bi]);
  }

  return results;
}

bool CollectionImpl::need_switch_to_new_segment() const {
  return writing_segment_->doc_count() >= schema_->max_doc_count_per_segment();
}

Status CollectionImpl::switch_to_new_segment_for_writing(
    const CollectionSchema::Ptr &schema) {
  auto s = writing_segment_->dump();
  CHECK_RETURN_STATUS(s);

  s = segment_manager_->add_segment(writing_segment_);
  CHECK_RETURN_STATUS(s);

  // when create new segment, segment meta should create a first new block
  // meta
  auto new_segment = Segment::CreateAndOpen(
      path_, schema == nullptr ? *schema_ : *schema, allocate_segment_id(),
      writing_segment_->meta()->max_doc_id() + 1, id_map_, delete_store_,
      version_manager_,
      SegmentOptions{false, options_.enable_mmap_, options_.max_buffer_size_,
                     options_.wal_durability_});
  if (!new_segment) {
    return new_segment.error();
  }

  Version version = version_manager_->get_current_version();
  auto writing_segment_meta = writing_segment_->meta();
  writing_segment_meta->remove_writing_forward_block();
  s = version.add_persisted_segment_meta(writing_segment_meta);
  CHECK_RETURN_STATUS(s);

  writing_segment_ = new_segment.value();
  version.reset_writing_segment_meta(writing_segment_->meta());
  version.set_next_segment_id(segment_id_allocator_.load());

  s = version_manager_->apply(version);
  CHECK_RETURN_STATUS(s);
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  return Status::OK();
}

Result<WriteResults> CollectionImpl::Delete(
    const std::vector<std::string> &pks) {
  CHECK_READONLY_RETURN_STATUS_EXPECTED();

  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  // TODO: The granularity of the write_lock is too coarse.
  std::lock_guard write_lock(write_mtx_);

  // Group-commit batched delete via the segment's DeleteBatch path: one
  // seg_mtx_ acquisition for the whole batch, one WAL group-commit
  // fsync at the end (under PER_BATCH durability).
  return writing_segment_->DeleteBatch(pks);
}

Status CollectionImpl::DeleteByFilter(const std::string &filter) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  auto segments = get_all_segments();

  VectorQuery query;
  query.filter_ = filter;
  query.topk_ = INT32_MAX;
  query.output_fields_ = std::vector<std::string>{};
  query.include_doc_id_ = true;

  auto ret = sql_engine_->execute(schema_, query, get_all_segments());
  if (!ret.has_value()) {
    return ret.error();
  }

  // TODO: The granularity of the write_lock is too coarse.
  std::lock_guard write_lock(write_mtx_);
  for (auto &doc : ret.value()) {
    Status s = writing_segment_->Delete(doc->doc_id());
    if (!s.ok()) {
      LOG_ERROR("Delete doc_id: %zu failed", (size_t)doc->doc_id());
      return s;
    }
  }

  return Status::OK();
}

Result<DocPtrList> CollectionImpl::Query(const VectorQuery &query) const {
  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  auto s = query.validate(schema_->get_vector_field(query.field_name_));
  CHECK_RETURN_STATUS_EXPECTED(s);

  auto segments = get_all_segments();
  if (segments.empty()) {
    return DocPtrList();
  }

  return sql_engine_->execute(schema_, query, segments);
}

Result<DocPtrList> CollectionImpl::QueryText(const TextQuery &query) const {
  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  // Validate against the (forward) field schema.
  auto *field_schema = schema_->get_forward_field(query.field_name_);
  auto vs = query.validate(field_schema);
  CHECK_RETURN_STATUS_EXPECTED(vs);

  auto segments = get_all_segments();
  if (segments.empty() || query.topk_ <= 0) {
    return DocPtrList();
  }

  const auto fts_op =
      query.op_ == TextQuery::MatchOp::AND ? FtsColumnIndexer::MatchOp::AND
                                           : FtsColumnIndexer::MatchOp::OR;
  const std::size_t per_segment_topk = static_cast<std::size_t>(query.topk_);

  // Per-segment search → (segment_ptr, global_doc_id, score) tuples.
  struct GlobalHit {
    Segment::Ptr segment;
    uint64_t global_doc_id;
    float score;
  };
  std::vector<GlobalHit> hits;

  for (const auto &segment : segments) {
    if (!segment) {
      continue;
    }
    auto fts = segment->get_fts_indexer(query.field_name_);
    if (!fts) {
      continue;  // segment has no FTS indexer for this field
    }
    auto search_result = fts->Search(query.text_, per_segment_topk, fts_op);
    if (!search_result.has_value()) {
      return tl::make_unexpected(search_result.error());
    }
    auto res = search_result.value();
    if (!res || res->count() == 0) {
      continue;
    }
    auto it = res->create_iterator();
    while (it->valid()) {
      const auto local_id = static_cast<uint32_t>(it->doc_id());
      const auto score = it->score();
      it->next();

      auto g = segment->get_global_doc_id(local_id);
      if (!g.has_value()) {
        continue;  // stale local id
      }
      const uint64_t global_doc_id = g.value();
      if (delete_store_->is_deleted(global_doc_id)) {
        continue;
      }
      hits.push_back(GlobalHit{segment, global_doc_id, score});
    }
  }

  if (hits.empty()) {
    return DocPtrList();
  }

  const std::size_t k = std::min(per_segment_topk, hits.size());
  std::partial_sort(hits.begin(), hits.begin() + k, hits.end(),
                    [](const GlobalHit &a, const GlobalHit &b) {
                      if (a.score != b.score) {
                        return a.score > b.score;
                      }
                      return a.global_doc_id < b.global_doc_id;
                    });
  hits.resize(k);

  // Materialize full Doc objects, attach BM25 scores, and (if requested)
  // narrow to the caller-specified output_fields.
  DocPtrList results;
  results.reserve(hits.size());
  for (const auto &h : hits) {
    auto doc = h.segment->Fetch(h.global_doc_id);
    if (!doc) {
      continue;
    }
    doc->set_score(h.score);
    if (query.output_fields_.has_value()) {
      const auto &keep = query.output_fields_.value();
      std::unordered_set<std::string> keep_set(keep.begin(), keep.end());
      for (const auto &fname : doc->field_names()) {
        if (keep_set.find(fname) == keep_set.end()) {
          doc->remove(fname);
        }
      }
    }
    results.push_back(doc);
  }
  return results;
}

Result<GroupResults> CollectionImpl::GroupByQuery(
    const GroupByVectorQuery &query) const {
  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  auto segments = get_all_segments();
  if (segments.empty()) {
    return GroupResults();
  }

  return sql_engine_->execute_group_by(schema_, query, segments);
}

Result<DocPtrMap> CollectionImpl::Fetch(
    const std::vector<std::string> &pks) const {
  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  auto segments = get_all_segments();

  DocPtrMap results;

  for (auto &pk : pks) {
    uint64_t doc_id;
    bool has = id_map_->has(pk, &doc_id);
    if (!has) {
      results.insert({pk, nullptr});
      continue;
    }
    if (delete_store_->is_deleted(doc_id)) {
      results.insert({pk, nullptr});
      continue;
    }
    auto segment = local_segment_by_doc_id(doc_id, segments);
    if (!segment) {
      LOG_WARN("doc_id: %zu segment not found", (size_t)doc_id);
      results.insert({pk, nullptr});
      continue;
    }
    results.insert({pk, segment->Fetch(doc_id)});
  }

  return results;
}

Status CollectionImpl::recovery() {
  if (!FileHelper::DirectoryExists(path_.c_str())) {
    return Status::InvalidArgument("collection path{", path_, "} not exist.");
  }

  // get lock file
  auto s = acquire_file_lock(false);
  CHECK_RETURN_STATUS(s);

  // recovery version first
  auto version_manager = VersionManager::Recovery(path_);
  if (!version_manager.has_value()) {
    return version_manager.error();
  }

  version_manager_ = version_manager.value();
  const auto v = version_manager_->get_current_version();
  schema_ = std::make_shared<CollectionSchema>(v.schema());
  options_.enable_mmap_ = v.enable_mmap();
  s = recover_idmap_and_delete_store();
  CHECK_RETURN_STATUS(s);

  // recover persist segments
  segment_manager_ = std::make_shared<SegmentManager>();

  auto segment_metas = v.persisted_segment_metas();

  SegmentOptions seg_options;
  seg_options.read_only_ = true;
  seg_options.enable_mmap_ = options_.enable_mmap_;
  for (size_t i = 0; i < segment_metas.size(); ++i) {
    auto segment = Segment::Open(path_, *schema_, *segment_metas[i], id_map_,
                                 delete_store_, version_manager_, seg_options);
    if (!segment) {
      return segment.error();
    }

    segment_manager_->add_segment(segment.value());
  }

  seg_options.read_only_ = options_.read_only_;
  seg_options.max_buffer_size_ = options_.max_buffer_size_;

  // recover writing segment
  auto writing_segment =
      Segment::Open(path_, *schema_, *v.writing_segment_meta(), id_map_,
                    delete_store_, version_manager_, seg_options);
  if (!writing_segment) {
    return writing_segment.error();
  }

  writing_segment_ = writing_segment.value();
  segment_id_allocator_.store(v.next_segment_id());

  // recover id map & delete store
  return Status::OK();
}

Status CollectionImpl::recover_idmap_and_delete_store() {
  const auto v = version_manager_->get_current_version();

  // idmap
  std::string idmap_path =
      FileHelper::MakeFilePath(path_, FileID::ID_FILE, v.id_map_path_suffix());
  id_map_ = IDMap::CreateAndOpen(schema_->name(), idmap_path, false,
                                 options_.read_only_);
  if (!id_map_) {
    return Status::InternalError("recovery idmap failed, path: ", idmap_path);
  }

  // delete store
  std::string delete_store_path = FileHelper::MakeFilePath(
      path_, FileID::DELETE_FILE, v.delete_snapshot_path_suffix());
  delete_store_ =
      DeleteStore::CreateAndLoad(schema_->name(), delete_store_path);
  if (!delete_store_) {
    return Status::InternalError("recovery delete store failed, path: ",
                                 delete_store_path);
  }

  return Status::OK();
}

Status CollectionImpl::create() {
  // check path
  if (path_.empty()) {
    return Status::InvalidArgument("path validate failed: path is empty");
  }
  if (!std::regex_match(path_, COLLECTION_PATH_REGEX)) {
    return Status::InvalidArgument("path validate failed: path[", path_,
                                   "] cannot pass the regex verification");
  }
  if (ailego::FileHelper::IsExist(path_.c_str())) {
    return Status::InvalidArgument("path validate failed: path[", path_,
                                   "] exists");
  }

  // check schema
  auto s = schema_->validate();
  CHECK_RETURN_STATUS(s);

  if (!ailego::FileHelper::MakePath(path_.c_str())) {
    return Status::InvalidArgument("create collection path failed: ", path_,
                                   ", error: ", strerror(errno));
  }

  // init lock file
  s = acquire_file_lock(true);
  CHECK_RETURN_STATUS(s);

  // init idmap & delete store
  s = create_idmap_and_delete_store();
  CHECK_RETURN_STATUS(s);

  // init version manager
  s = init_version_manager();
  CHECK_RETURN_STATUS(s);

  // create segment
  s = init_writing_segment();
  CHECK_RETURN_STATUS(s);

  // init version
  Version version;
  version.set_schema(*schema_);
  version.set_enable_mmap(options_.enable_mmap_);
  version.reset_writing_segment_meta(writing_segment_->meta());
  version.set_id_map_path_suffix(0);
  version.set_delete_snapshot_path_suffix(0);
  version.set_next_segment_id(1);

  version_manager_->apply(version);
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  segment_id_allocator_.store(1);
  segment_manager_ = std::make_unique<SegmentManager>();

  return Status::OK();
}

Status CollectionImpl::create_idmap_and_delete_store() {
  // idmap
  std::string idmap_path = FileHelper::MakeFilePath(path_, FileID::ID_FILE, 0);
  id_map_ = IDMap::CreateAndOpen(schema_->name(), idmap_path, true,
                                 options_.read_only_);
  if (!id_map_) {
    return Status::InternalError("create id map failed, path: ", idmap_path);
  }

  std::string delete_store_path =
      FileHelper::MakeFilePath(path_, FileID::DELETE_FILE, 0);
  delete_store_ = std::make_shared<DeleteStore>(schema_->name());
  // when first create collection, delete store will flush a empty snapshot
  delete_store_->flush(delete_store_path);

  return Status::OK();
}

Status CollectionImpl::init_version_manager() {
  // use empty version to init version manager
  auto version_manager = VersionManager::Create(path_, Version{});
  if (!version_manager.has_value()) {
    return version_manager.error();
  }

  version_manager_ = version_manager.value();
  return Status::OK();
}

Status CollectionImpl::init_writing_segment() {
  SegmentOptions options;
  options.enable_mmap_ = options_.enable_mmap_;
  options.max_buffer_size_ = options_.max_buffer_size_;
  options.read_only_ = options_.read_only_;

  auto writing_segment = Segment::CreateAndOpen(
      path_, *schema_, 0, 0, id_map_, delete_store_, version_manager_, options);

  if (!writing_segment) {
    return writing_segment.error();
  }

  writing_segment_ = writing_segment.value();

  return Status::OK();
}

Status CollectionImpl::acquire_file_lock(bool create) {
  std::string lock_file_path = ailego::StringHelper::Concat(path_, "/", "LOCK");

  if (create) {
    if (!lock_file_.create(lock_file_path.c_str(), 0)) {
      return Status::InternalError("Can't create lock file: ", lock_file_path);
    }
  } else {
    if (!lock_file_.open(lock_file_path.c_str(), false)) {
      return Status::InternalError("Can't open lock file: ", lock_file_path);
    }
  }

  if (options_.read_only_) {
    if (!ailego::FileLock::TryLockShared(lock_file_.native_handle())) {
      return Status::InternalError("Can't lock read-only collection: ",
                                   lock_file_path);
    }
  } else {
    if (!ailego::FileLock::TryLock(lock_file_.native_handle())) {
      return Status::InternalError("Can't lock read-write collection: ",
                                   lock_file_path);
    }
  }

  return Status::OK();
}

Segment::Ptr CollectionImpl::local_segment_by_doc_id(
    uint64_t doc_id, const std::vector<Segment::Ptr> &segments) const {
  size_t left = 0;
  size_t right = segments.size();

  while (left < right) {
    size_t mid = left + (right - left) / 2;
    uint64_t min_id = segments[mid]->meta()->min_doc_id();
    uint64_t max_id = segments[mid]->meta()->max_doc_id();

    if (doc_id < min_id) {
      right = mid;
    } else if (doc_id > max_id) {
      left = mid + 1;
    } else {
      return segments[mid];
    }
  }

  return nullptr;
}

std::vector<Segment::Ptr> CollectionImpl::get_all_segments() const {
  // Persisted segments first, then every non-empty writing shard. The
  // loop is trivially one iteration under Phase 4 (N_shards = 1) but
  // is the Phase-5 fan-out point — enumerating every shard's writing
  // segment will "just work" once all_writing_segments() grows.
  std::vector<Segment::Ptr> segments = get_all_persist_segments();
  for (const auto &ws : all_writing_segments()) {
    if (ws && ws->doc_count() > 0) {
      segments.push_back(ws);
    }
  }
  return segments;
}

std::vector<Segment::Ptr> CollectionImpl::get_all_persist_segments() const {
  return segment_manager_->get_segments();
}

}  // namespace zvec
