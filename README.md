<p align="right">
  English | <a href="./README_CN.md">简体中文</a>
</p>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_log_2.svg" />
    <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_logo_1.svg" width="400" alt="zvec logo" />
  </picture>
</div>

<p align="center">
  <a href="https://codecov.io/github/alibaba/zvec"><img src="https://codecov.io/github/alibaba/zvec/graph/badge.svg?token=O81CT45B66" alt="Code Coverage"/></a>
  <a href="https://github.com/alibaba/zvec/actions/workflows/01-ci-pipeline.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/01-ci-pipeline.yml/badge.svg?branch=main" alt="Main"/></a>
  <a href="https://github.com/alibaba/zvec/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/pypi/v/zvec.svg" alt="PyPI Release"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/badge/python-3.10%20~%203.14-blue.svg" alt="Python Versions"/></a>
  <a href="https://www.npmjs.com/package/@zvec/zvec"><img src="https://img.shields.io/npm/v/@zvec/zvec.svg" alt="npm Release"/></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/20830" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20830" alt="alibaba%2Fzvec | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://zvec.org/en/docs/quickstart/">🚀 <strong>Quickstart</strong> </a> |
  <a href="https://zvec.org/en/">🏠 <strong>Home</strong> </a> |
  <a href="https://zvec.org/en/docs/">📚 <strong>Docs</strong> </a> |
  <a href="https://zvec.org/en/docs/benchmarks/">📊 <strong>Benchmarks</strong> </a> |
  <a href="https://deepwiki.com/alibaba/zvec">🔎 <strong>DeepWiki</strong> </a> |
  <a href="https://discord.gg/rKddFBBu9z">🎮 <strong>Discord</strong> </a> |
  <a href="https://x.com/ZvecAI">🐦 <strong>X (Twitter)</strong> </a>
</p>

**Zvec** is an open-source, in-process vector database — lightweight, lightning-fast, and designed to embed directly into applications. Built on **Proxima** (Alibaba's battle-tested vector search engine), it delivers production-grade, low-latency, scalable similarity search with minimal setup.

> [!IMPORTANT]
> **🚀 v0.3.0 Released on April 3, 2026**
>
> - **New Platforms**: Initial **Windows (MSVC)** and **Android** support. Published official Windows **Python** and **Node.js** packages.
> - **Efficiency**: **RabitQ** quantization and **CPU Auto-Dispatch** for optimized SIMD execution.
> - **Ecosystem**: **C-API** for custom language bindings and **[MCP](https://github.com/zvec-ai/zvec-mcp-server) / [Skill](https://github.com/zvec-ai/zvec-agent-skills)** integration for AI Agents.
>
> 👉 [Read the Release Notes](https://github.com/alibaba/zvec/releases/tag/v0.3.0) | [View Roadmap 📍](https://github.com/alibaba/zvec/issues/309)

## 💫 Features

- **Blazing Fast**: Searches billions of vectors in milliseconds.
- **Simple, Just Works**: [Install](#-installation) and start searching in seconds. No servers, no config, no fuss.
- **Dense + Sparse Vectors**: Work with both dense and sparse embeddings, with native support for multi-vector queries in a single call.
- **Hybrid Search**: Combine semantic similarity with [Full-Text Search (BM25)](#-full-text-search-bm25) and structured filters for precise results.
- **Write Sharding**: [Parallel writers](#-write-sharding--concurrency) across multiple shards with per-shard locking for high-throughput ingestion.
- **Background Optimization**: [AutoOptimizer](#autooptimizer) merges segments automatically; [OptimizeOption](#manual-optimization) gives full control over parallelism, memory budgets, and cancellation.
- **Runs Anywhere**: As an in-process library, Zvec runs wherever your code runs — notebooks, servers, CLI tools, or even edge devices.

## 📦 Installation

### [Python](https://pypi.org/project/zvec/)

**Requirements**: Python 3.10 - 3.14

```bash
pip install zvec
```

### [Node.js](https://www.npmjs.com/package/@zvec/zvec)

```bash
npm install @zvec/zvec
```

### ✅ Supported Platforms

- Linux (x86_64, ARM64)
- macOS (ARM64)
- Windows (x86_64)

### 🛠️ Building from Source

If you prefer to build Zvec from source, please check the [Building from Source](https://zvec.org/en/docs/build/) guide.

## ⚡ One-Minute Example

```python
import zvec

# Define collection schema
schema = zvec.CollectionSchema(
    name="example",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 4),
)

# Create collection
collection = zvec.create_and_open(path="./zvec_example", schema=schema)

# Insert documents
collection.insert([
    zvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]}),
    zvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.1]}),
])

# Search by vector similarity
results = collection.query(
    zvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]),
    topk=10
)

# Results: list of {'id': str, 'score': float, ...}, sorted by relevance
print(results)
```

## 🔧 Configuration

Call `zvec.init()` once before any operation. All parameters are optional — omit them to let Zvec auto-detect from the runtime environment (cgroup-aware for containers).

```python
import zvec
from zvec import LogLevel, LogType

zvec.init(
    log_type=LogType.CONSOLE,              # CONSOLE or FILE
    log_level=LogLevel.WARN,               # DEBUG, INFO, WARN, ERROR, FATAL
    query_threads=None,                    # None = auto-detect from CPU/cgroup
    optimize_threads=None,                 # threads for background optimization
    max_query_topk=1024,                   # max allowed topk per query (default 1024)
    memory_limit_mb=None,                  # soft memory cap (None = cgroup * 0.8)
    invert_to_forward_scan_ratio=0.9,      # cost-based optimizer threshold [0, 1]
    brute_force_by_keys_ratio=0.1,         # brute-force vs index threshold [0, 1]
)
```

## 🔍 Full-Text Search (BM25)

Add keyword search alongside vector similarity using the built-in BM25 index:

```python
import zvec

schema = zvec.CollectionSchema(
    name="articles",
    fields=[
        zvec.FieldSchema("title", zvec.DataType.STRING),
        zvec.FieldSchema(
            "body", zvec.DataType.STRING,
            index_param=zvec.FtsIndexParam(tokenizer="default", k1=1.2, b=0.75),
        ),
    ],
    vectors=[zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 128)],
)
collection = zvec.create_and_open("./articles_db", schema)

# Insert documents
collection.insert([
    zvec.Doc(id="1", fields={"title": "Intro", "body": "the quick brown fox"},
             vectors={"embedding": [0.1] * 128}),
    zvec.Doc(id="2", fields={"title": "Other", "body": "a lazy dog sleeps"},
             vectors={"embedding": [0.2] * 128}),
])

# Text-only search
results = collection.query_text(
    zvec.TextQuery(field_name="body", text="quick fox", topk=10, op=zvec.MatchOp.OR)
)

# Hybrid search (vector + BM25 fused with RRF)
results = collection.query(
    vectors=zvec.VectorQuery("embedding", vector=[0.1] * 128),
    text=zvec.TextQuery(field_name="body", text="quick fox", topk=10),
    topk=10,
    reranker=zvec.RrfReRanker(topn=10),
)
```

`MatchOp.OR` matches docs containing **any** query term; `MatchOp.AND` requires **all** terms.

## ⚡ Write Sharding & Concurrency

Enable write-side sharding for parallel ingestion from multiple threads:

```python
option = zvec.CollectionOption(
    write_shards=4,                          # 4 independent write segments
    wal_durability=zvec.WalDurability.PER_BATCH,  # one fsync per batch (default)
)
collection = zvec.create_and_open("./sharded_db", schema, option=option)
```

Each shard has its own mutex — writers targeting different shards run concurrently. Documents are routed to shards by a deterministic CRC32C hash of the primary key.

**WAL Durability Modes:**

| Mode | Behavior | Use Case |
|------|----------|----------|
| `WalDurability.NONE` | No fsync; OS flushes eventually | Maximum throughput, crash-unsafe |
| `WalDurability.PER_BATCH` | One fsync per write batch (default) | Balanced durability/speed |
| `WalDurability.PER_DOC` | fsync after every record | Strongest durability |

## 🔄 Optimization

### Manual Optimization

Merge segments and rebuild indices with full control over parallelism:

```python
# Basic optimization
collection.optimize()

# Advanced: parallel dispatch with memory budget and cancellation
token = zvec.CancelToken()
collection.optimize(zvec.OptimizeOption(
    concurrency=4,                     # threads per compact/index task (0 = auto)
    parallel_tasks=2,                  # max concurrent tasks (0 = auto)
    memory_budget_bytes=4 * 1024**3,   # 4 GB soft limit across tasks
    cancel_token=token,                # cooperative cancellation
))

# Cancel from another thread if needed
token.cancel()
```

### AutoOptimizer

Enable background optimization that runs automatically when segments accumulate:

```python
option = zvec.CollectionOption(
    auto_optimize_enabled=True,
    auto_optimize_interval_seconds=60,     # check every 60s (default)
    auto_optimize_max_segments=10,         # trigger when > 10 segments (default)
    auto_optimize_cooldown_seconds=300,    # min 5 min between runs (default)
)
collection = zvec.create_and_open("./auto_opt_db", schema, option=option)
# The background thread monitors segment count and optimizes automatically.
```

## 📋 Index Types

| Index | Class | Best For |
|-------|-------|----------|
| **HNSW** | `HnswIndexParam(m=50, ef_construction=500)` | General-purpose ANN search |
| **HNSW-RabitQ** | `HnswRabitqIndexParam(total_bits=7, num_clusters=16)` | Memory-efficient ANN with quantization |
| **IVF** | `IVFIndexParam(n_list=0, n_iters=10)` | Large-scale with cluster-based pruning |
| **Flat** | `FlatIndexParam()` | Exact search (brute-force) |
| **Inverted** | `InvertIndexParam(enable_range_optimization=True)` | Scalar field filtering |
| **FTS** | `FtsIndexParam(tokenizer="default", k1=1.2, b=0.75)` | Full-text keyword search (BM25) |

**Query-Time Parameters:**

```python
# HNSW: control accuracy vs speed tradeoff
param = zvec.HnswQueryParam(ef=300, is_using_refiner=False)
results = collection.query(
    zvec.VectorQuery("embedding", vector=[...], param=param),
    topk=10,
)

# IVF: control cluster probing depth
param = zvec.IVFQueryParam(nprobe=20)
```

## 📈 Performance at Scale

Zvec delivers exceptional speed and efficiency, making it ideal for demanding production workloads.

<img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qps_10M.svg" width="800" alt="Zvec Performance Benchmarks" />

For detailed benchmark methodology, configurations, and complete results, please see our [Benchmarks documentation](https://zvec.org/en/docs/benchmarks/).

## 🤝 Join Our Community

<div align="center">

Stay updated and get support — scan or click:

<div align="center">

| 💬 DingTalk | 📱 WeChat | 🎮 Discord | X (Twitter) |
| :---: | :---: | :---: | :---: |
| <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/dingding.png" width="150" alt="DingTalk QR Code"/> | <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/wechat.png?v=5" width="150" alt="WeChat QR Code"/> | [![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/rKddFBBu9z) | [![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/ZvecAI)](<https://x.com/ZvecAI>) |
| Scan to join | Scan to join | Click to join | Click to follow |

</div>

</div>

## ❤️ Contributing

We welcome and appreciate contributions from the community! Whether you're fixing a bug, adding a feature, or improving documentation, your help makes Zvec better for everyone.

Check out our [Contributing Guide](./CONTRIBUTING.md) to get started!
