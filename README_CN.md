<p align="right">
  <a href="./README.md">English</a> | 简体中文
</p>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_log_2.svg" />
    <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_logo_1.svg" width="400" alt="zvec logo" />
  </picture>
</div>

<p align="center">
  <a href="https://codecov.io/github/alibaba/zvec"><img src="https://codecov.io/github/alibaba/zvec/graph/badge.svg?token=O81CT45B66" alt="代码覆盖率"/></a>
  <a href="https://github.com/alibaba/zvec/actions/workflows/01-ci-pipeline.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/01-ci-pipeline.yml/badge.svg?branch=main" alt="Main"/></a>
  <a href="https://github.com/alibaba/zvec/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="许可证"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/pypi/v/zvec.svg" alt="PyPI 版本"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/badge/python-3.10%20~%203.14-blue.svg" alt="Python 版本"/></a>
  <a href="https://www.npmjs.com/package/@zvec/zvec"><img src="https://img.shields.io/npm/v/@zvec/zvec.svg" alt="npm 版本"/></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/20830" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20830" alt="alibaba%2Fzvec | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://zvec.org/en/docs/quickstart/">🚀 <strong>快速开始</strong> </a> |
  <a href="https://zvec.org/en/">🏠 <strong>主页</strong> </a> |
  <a href="https://zvec.org/en/docs/">📚 <strong>文档</strong> </a> |
  <a href="https://zvec.org/en/docs/benchmarks/">📊 <strong>性能报告</strong> </a> |
  <a href="https://deepwiki.com/alibaba/zvec">🔎 <strong>DeepWiki</strong> </a> |
  <a href="https://discord.gg/rKddFBBu9z">🎮 <strong>Discord</strong> </a> |
  <a href="https://x.com/ZvecAI">🐦 <strong>X (Twitter)</strong> </a>
</p>

**Zvec** 是一款开源的嵌入式(进程内)向量数据库 — 轻量、极速，可直接嵌入应用程序。以极简的配置提供生产级、低延迟、可扩展的向量检索能力。

> [!IMPORTANT]
> **🚀 v0.3.0 已于 2026 年 4 月 3 日发布**
>
> - **新平台支持**：支持 **Windows (MSVC)** 和 **Android**。发布了官方 Windows **Python** 和 **Node.js** 安装包。
> - **性能优化**：集成 **RabitQ** 量化以及 **CPU 指令集自适应检测**，优化 SIMD 执行。
> - **生态集成**：提供 **C-API** 用于多种编程语言绑定，以及 **[MCP](https://github.com/zvec-ai/zvec-mcp-server) / [Skill](https://github.com/zvec-ai/zvec-agent-skills)** 集成。
>
> 👉 [查看发布说明](https://github.com/alibaba/zvec/releases/tag/v0.3.0) | [查看路线图 📍](https://github.com/alibaba/zvec/issues/309)

## 💫 核心特性

- **极致性能**：毫秒级响应，轻松检索数十亿级向量。
- **开箱即用**：[安装](#-安装)后即刻开始搜索，无需服务器、无需配置、零门槛。
- **稠密 + 稀疏向量**：支持稠密向量和稀疏向量，提供多向量联合查询的原生支持。
- **混合检索**：向量语义搜索 + [全文搜索 (BM25)](#-全文搜索-bm25) + 标量条件过滤，获得精确结果。
- **写入分片**：[多分片并行写入](#-写入分片与并发)，每个分片独立加锁，实现高吞吐量数据写入。
- **后台优化**：[AutoOptimizer](#自动优化) 自动合并段；[OptimizeOption](#手动优化) 提供并行度、内存预算和取消控制。
- **进程内运行**：无需单独部署服务，纯进程内运行。Notebook、高性能服务器、CLI 工具、边缘设备 — 随处可用。

## 📦 安装

### [Python](https://pypi.org/project/zvec/)

**环境要求**：Python 3.10 - 3.14

```bash
pip install zvec
```

### [Node.js](https://www.npmjs.com/package/@zvec/zvec)

```bash
npm install @zvec/zvec
```

### ✅ 支持的平台

- Linux (x86_64, ARM64)
- macOS (ARM64)
- Windows (x86_64)

### 🛠️ 源码构建

如需从源码构建 Zvec，请参考[源码构建指南](https://zvec.org/en/docs/build/)。

## ⚡ 一分钟上手

```python
import zvec

# 定义 collection schema
schema = zvec.CollectionSchema(
    name="example",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 4),
)

# 创建 collection
collection = zvec.create_and_open(path="./zvec_example", schema=schema)

# 插入 documents
collection.insert([
    zvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]}),
    zvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.1]}),
])

# 向量相似度检索
results = collection.query(
    zvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]),
    topk=10
)

# 查询结果：按相关性排序的 {'id': str, 'score': float, ...} 列表
print(results)
```

## 🔧 配置

在执行任何操作之前调用一次 `zvec.init()`。所有参数均为可选 — 省略时 Zvec 会从运行环境自动检测（支持 cgroup 容器感知）。

```python
import zvec
from zvec import LogLevel, LogType

zvec.init(
    log_type=LogType.CONSOLE,              # CONSOLE 或 FILE
    log_level=LogLevel.WARN,               # DEBUG, INFO, WARN, ERROR, FATAL
    query_threads=None,                    # None = 从 CPU/cgroup 自动检测
    optimize_threads=None,                 # 后台优化线程数
    max_query_topk=1024,                   # 每次查询允许的最大 topk（默认 1024）
    memory_limit_mb=None,                  # 软内存上限（None = cgroup * 0.8）
    invert_to_forward_scan_ratio=0.9,      # 基于代价的优化器阈值 [0, 1]
    brute_force_by_keys_ratio=0.1,         # 暴力搜索 vs 索引阈值 [0, 1]
)
```

## 🔍 全文搜索 (BM25)

通过内置 BM25 索引在向量相似度检索的基础上增加关键词搜索：

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

# 插入文档
collection.insert([
    zvec.Doc(id="1", fields={"title": "简介", "body": "the quick brown fox"},
             vectors={"embedding": [0.1] * 128}),
    zvec.Doc(id="2", fields={"title": "其他", "body": "a lazy dog sleeps"},
             vectors={"embedding": [0.2] * 128}),
])

# 纯文本搜索
results = collection.query_text(
    zvec.TextQuery(field_name="body", text="quick fox", topk=10, op=zvec.MatchOp.OR)
)

# 混合搜索（向量 + BM25，通过 RRF 融合）
results = collection.query(
    vectors=zvec.VectorQuery("embedding", vector=[0.1] * 128),
    text=zvec.TextQuery(field_name="body", text="quick fox", topk=10),
    topk=10,
    reranker=zvec.RrfReRanker(topn=10),
)
```

`MatchOp.OR` 匹配包含**任意**查询词的文档；`MatchOp.AND` 要求**所有**查询词都必须出现。

## ⚡ 写入分片与并发

启用写入侧分片以支持多线程并行写入：

```python
option = zvec.CollectionOption(
    write_shards=4,                          # 4 个独立写入段
    wal_durability=zvec.WalDurability.PER_BATCH,  # 每批次一次 fsync（默认）
)
collection = zvec.create_and_open("./sharded_db", schema, option=option)
```

每个分片有独立的互斥锁 — 写入不同分片的线程可以并发执行。文档通过主键的 CRC32C 哈希确定性路由到对应分片。

**WAL 持久性模式：**

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| `WalDurability.NONE` | 不主动 fsync；由操作系统决定何时刷盘 | 最大吞吐量，崩溃不安全 |
| `WalDurability.PER_BATCH` | 每个写入批次一次 fsync（默认） | 平衡持久性与速度 |
| `WalDurability.PER_DOC` | 每条记录后立即 fsync | 最强持久性保证 |

## 🔄 优化

### 手动优化

合并段并重建索引，支持完整的并行度控制：

```python
# 基础优化
collection.optimize()

# 高级：并行分发 + 内存预算 + 取消支持
token = zvec.CancelToken()
collection.optimize(zvec.OptimizeOption(
    concurrency=4,                     # 每个压缩/索引任务的线程数（0 = 自动）
    parallel_tasks=2,                  # 最大并发任务数（0 = 自动）
    memory_budget_bytes=4 * 1024**3,   # 4 GB 软限制
    cancel_token=token,                # 协作式取消
))

# 可从其他线程取消
token.cancel()
```

### 自动优化

启用后台自动优化，当段数量积累时自动触发：

```python
option = zvec.CollectionOption(
    auto_optimize_enabled=True,
    auto_optimize_interval_seconds=60,     # 每 60 秒检查一次（默认）
    auto_optimize_max_segments=10,         # 超过 10 个段时触发（默认）
    auto_optimize_cooldown_seconds=300,    # 两次运行间至少 5 分钟（默认）
)
collection = zvec.create_and_open("./auto_opt_db", schema, option=option)
# 后台线程自动监控段数量并执行优化。
```

## 📋 索引类型

| 索引 | 类 | 适用场景 |
|------|---|----------|
| **HNSW** | `HnswIndexParam(m=50, ef_construction=500)` | 通用近似最近邻搜索 |
| **HNSW-RabitQ** | `HnswRabitqIndexParam(total_bits=7, num_clusters=16)` | 内存高效的量化 ANN |
| **IVF** | `IVFIndexParam(n_list=0, n_iters=10)` | 基于聚类剪枝的大规模检索 |
| **Flat** | `FlatIndexParam()` | 精确搜索（暴力扫描） |
| **Inverted** | `InvertIndexParam(enable_range_optimization=True)` | 标量字段过滤 |
| **FTS** | `FtsIndexParam(tokenizer="default", k1=1.2, b=0.75)` | 全文关键词搜索 (BM25) |

**查询时参数：**

```python
# HNSW：控制精度与速度的权衡
param = zvec.HnswQueryParam(ef=300, is_using_refiner=False)
results = collection.query(
    zvec.VectorQuery("embedding", vector=[...], param=param),
    topk=10,
)

# IVF：控制聚类探测深度
param = zvec.IVFQueryParam(nprobe=20)
```

## 📈 极致性能

Zvec 提供极致的速度和效率，能够轻松应对高要求的生产环境负载。

<img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qps_10M.svg" width="800" alt="Zvec 性能基准测试" />

有关具体的测试方法、配置及完整结果，请参阅[性能报告](https://zvec.org/en/docs/benchmarks/)。

## 🤝 加入社区

<div align="center">

获取最新动态和技术支持：

<div align="center">

| 💬 钉钉群 | 📱 微信群 | 🎮 Discord | X (Twitter) |
| :---: | :---: | :---: | :---: |
| <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/dingding.png" width="150" alt="钉钉二维码"/> | <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/wechat.png?v=5" width="150" alt="微信二维码"/> | [![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/rKddFBBu9z) | [![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/ZvecAI)](<https://x.com/ZvecAI>) |
| 扫码加入 | 扫码加入 | 点击加入 | 点击关注 |

</div>

</div>

## ❤️ 参与贡献

非常欢迎来自社区的每一份贡献！无论是修复 Bug、新增功能，还是完善文档，都将让 Zvec 变得更好。

请查阅我们的[贡献指南](./CONTRIBUTING.md)开始参与！
