#!/usr/bin/env python3
"""
Large-scale benchmark suite for zvec.

Tests every major feature at high scale with detailed topk sweeps:
  - Insert throughput (50K docs, single + sharded)
  - HNSW optimize + recall sweep (topk 10 → 20K)
  - Full-text search (BM25) correctness + throughput at scale
  - Hybrid search (vector + FTS + RRF)
  - Write sharding + concurrent writers at scale
  - WAL durability comparison
  - HNSW optimize multi-batch (large)
  - AutoOptimizer (HNSW, background)
  - CancelToken cooperative cancellation
  - Max topk enforcement
  - Upsert, delete, concurrent R/W at scale
  - Edge cases and defaults validation

Usage:
    python python/tests/benchmark_deep.py
"""

from __future__ import annotations

import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import zvec
from zvec import (
    CancelToken,
    CollectionOption,
    CollectionSchema,
    DataType,
    Doc,
    FieldSchema,
    FlatIndexParam,
    FtsIndexParam,
    HnswIndexParam,
    HnswQueryParam,
    IndexType,
    MatchOp,
    MetricType,
    OptimizeOption,
    RrfReRanker,
    TextQuery,
    VectorQuery,
    VectorSchema,
    WalDurability,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIM = 128
MAX_TOPK = 50_000
BATCH = 1000  # max insert batch (zvec limit is 1024)
_initialized = False


def ensure_init():
    global _initialized
    if not _initialized:
        try:
            zvec.init(log_level=zvec.LogLevel.WARN, max_query_topk=MAX_TOPK)
        except RuntimeError:
            pass
        _initialized = True


def rvecs(n: int, dim: int = DIM, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


@contextmanager
def tcoll(name, schema, option=None):
    ensure_init()
    tmpdir = tempfile.mkdtemp(prefix=f"zvec_{name}_")
    path = os.path.join(tmpdir, name)
    option = option or CollectionOption(enable_mmap=False)
    coll = zvec.create_and_open(path, schema, option=option)
    try:
        yield coll
    finally:
        try:
            coll.destroy()
        except Exception:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)


def hnsw_schema(dim=DIM, extra_fields=None):
    return CollectionSchema(
        name="bench", fields=extra_fields or [],
        vectors=[VectorSchema("vec", DataType.VECTOR_FP32, dim,
                              index_param=HnswIndexParam(metric_type=MetricType.IP,
                                                         m=32, ef_construction=400))])

def fts_hnsw_schema(dim=DIM):
    return CollectionSchema(
        name="bench_fts", fields=[
            FieldSchema("body", DataType.STRING, nullable=True,
                        index_param=FtsIndexParam(tokenizer="default", k1=1.2, b=0.75)),
            FieldSchema("category", DataType.STRING, nullable=True),
        ],
        vectors=[VectorSchema("vec", DataType.VECTOR_FP32, dim,
                              index_param=HnswIndexParam(metric_type=MetricType.IP,
                                                         m=32, ef_construction=400))])


def insert_batched(coll, docs):
    for s in range(0, len(docs), BATCH):
        statuses = coll.insert(docs[s:s + BATCH])
        for st in statuses:
            if not st.ok():
                raise RuntimeError(f"insert failed: {st.message()}")


@dataclass
class R:
    name: str
    elapsed_s: float = 0.0
    throughput: float = 0.0
    recall: float = 0.0
    details: dict = field(default_factory=dict)
    passed: bool = True
    error: str = ""


def fmt(r: R) -> str:
    tag = "PASS" if r.passed else "FAIL"
    parts = [f"[{tag}] {r.name}: {r.elapsed_s:.3f}s"]
    if r.throughput > 0:
        parts.append(f"  throughput={r.throughput:,.0f} ops/s")
    if r.recall > 0:
        parts.append(f"  recall={r.recall:.4f}")
    if r.error:
        parts.append(f"  error: {r.error}")
    for k, v in r.details.items():
        parts.append(f"  {k}={v}")
    return "\n".join(parts)


# ===================================================================
# 1. INSERT THROUGHPUT  (50K, single-threaded)
# ===================================================================

def bench_insert(n=50_000):
    vecs = rvecs(n)
    with tcoll("insert", hnsw_schema()) as c:
        docs = [Doc(id=f"d{i}", vectors={"vec": vecs[i].tolist()}) for i in range(n)]
        t0 = time.perf_counter()
        insert_batched(c, docs)
        dt = time.perf_counter() - t0
        if c.stats.doc_count != n:
            return R("insert_50k", error=f"count {c.stats.doc_count}!={n}", passed=False)
    return R("insert_50k", dt, throughput=n / dt, details={"n": n, "dim": DIM})


# ===================================================================
# 2. SHARDED INSERT  (50K, 4 shards, 4 threads)
# ===================================================================

def bench_sharded(n=50_000, shards=4, threads=4):
    vecs = rvecs(n)
    opt = CollectionOption(write_shards=shards, enable_mmap=False, wal_durability=WalDurability.NONE)
    with tcoll("sharded", hnsw_schema(), opt) as c:
        per = n // threads
        errs = []
        def w(tid):
            s, e = tid * per, (tid + 1) * per
            for bs in range(s, e, BATCH):
                be = min(bs + BATCH, e)
                batch = [Doc(id=f"t{tid}_d{i}", vectors={"vec": vecs[i].tolist()}) for i in range(bs, be)]
                for st in c.insert(batch):
                    if not st.ok():
                        errs.append(st.message())
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futs = [pool.submit(w, t) for t in range(threads)]
            for f in as_completed(futs): f.result()
        dt = time.perf_counter() - t0
        total = per * threads
        if errs:
            return R("sharded_50k", dt, error=f"{len(errs)} failures", passed=False)
        if c.stats.doc_count != total:
            return R("sharded_50k", dt, error=f"count {c.stats.doc_count}!={total}", passed=False)
    return R("sharded_50k", dt, throughput=total / dt,
             details={"n": total, "shards": shards, "threads": threads})


# ===================================================================
# 3. HNSW OPTIMIZE + RECALL SWEEP  (50K docs, topk 10→20K)
# ===================================================================

def bench_recall_sweep(n=50_000, n_queries=200):
    vecs = rvecs(n, seed=42)
    qvecs = rvecs(n_queries, seed=99)

    with tcoll("recall", hnsw_schema()) as c:
        docs = [Doc(id=f"d{i}", vectors={"vec": vecs[i].tolist()}) for i in range(n)]
        insert_batched(c, docs)

        t0 = time.perf_counter()
        c.optimize(OptimizeOption(concurrency=0, parallel_tasks=0))
        opt_sec = time.perf_counter() - t0

        # brute-force ground truth
        scores = qvecs @ vecs.T

        topk_values = [10, 100, 500, 1000, 2000, 5000, 10_000, 20_000]
        rows = []
        for topk in topk_values:
            actual_topk = min(topk, n)
            gt = np.argsort(-scores, axis=1)[:, :actual_topk]
            ef = min(max(topk, 400), n)
            lats = []
            recs = []
            for qi in range(n_queries):
                t0 = time.perf_counter()
                res = c.query(VectorQuery("vec", vector=qvecs[qi].tolist(),
                                          param=HnswQueryParam(ef=ef)), topk=topk)
                lats.append(time.perf_counter() - t0)
                found = {int(d.id[1:]) for d in res}
                truth = set(gt[qi].tolist())
                recs.append(len(found & truth) / actual_topk if actual_topk > 0 else 1.0)

            mean_lat = statistics.mean(lats)
            p99_lat = sorted(lats)[int(0.99 * len(lats))]
            mean_rec = statistics.mean(recs)
            returned = len(res)
            rows.append({
                "topk": topk, "returned": returned,
                "recall": f"{mean_rec:.4f}",
                "mean_ms": f"{mean_lat * 1000:.2f}",
                "p99_ms": f"{p99_lat * 1000:.2f}",
            })

        details = {"n_docs": n, "n_queries": n_queries, "optimize_sec": f"{opt_sec:.2f}"}
        for row in rows:
            k = row["topk"]
            details[f"topk={k}"] = f"recall={row['recall']}  returned={row['returned']}  mean={row['mean_ms']}ms  p99={row['p99_ms']}ms"

        worst_recall = min(float(r["recall"]) for r in rows)

    return R("recall_sweep_50k", opt_sec, recall=worst_recall, details=details)


# ===================================================================
# 4. FTS CORRECTNESS
# ===================================================================

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a fast red fox leaps across the sleeping hound",
    "machine learning models require large datasets",
    "neural networks are a subset of machine learning",
    "the lazy dog sleeps all day long",
    "foxes are clever animals found worldwide",
    "deep learning revolutionized natural language processing",
    "the brown bear roams through the dense forest",
    "artificial intelligence encompasses many subfields",
    "reinforcement learning enables agents to make decisions",
]

def bench_fts_correct():
    with tcoll("fts_c", fts_hnsw_schema()) as c:
        c.insert([Doc(id=f"d{i}", fields={"body": t, "category": "test"},
                      vectors={"vec": [0.0] * DIM}) for i, t in enumerate(CORPUS)])
        errs = []

        # OR
        res = c.query_text(TextQuery(field_name="body", text="fox", topk=10))
        if {d.id for d in res} != {"d0", "d1"}:
            errs.append(f"OR fox: {[d.id for d in res]}")
        # descending scores
        if len(res) >= 2 and res[0].score < res[1].score:
            errs.append("scores not descending")
        # AND
        res = c.query_text(TextQuery(field_name="body", text="quick fox", topk=10, op=MatchOp.AND))
        if {d.id for d in res} != {"d0"}:
            errs.append(f"AND quick fox: {[d.id for d in res]}")
        # topk
        res = c.query_text(TextQuery(field_name="body", text="the", topk=3))
        if len(res) > 3:
            errs.append(f"topk=3 got {len(res)}")
        # delete
        c.delete("d0")
        res = c.query_text(TextQuery(field_name="body", text="fox", topk=10))
        if "d0" in {d.id for d in res}:
            errs.append("deleted d0 returned")
        # unknown field
        try:
            c.query_text(TextQuery(field_name="nope", text="x", topk=10))
            errs.append("unknown field didn't raise")
        except Exception:
            pass

        if errs:
            return R("fts_correct", error="; ".join(errs), passed=False)
    return R("fts_correct")


# ===================================================================
# 5. FTS THROUGHPUT  (10K docs, 500 queries)
# ===================================================================

def bench_fts_throughput(n=10_000, nq=500):
    rng = np.random.default_rng(123)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliet", "kilo", "lima"]
    with tcoll("fts_tp", fts_hnsw_schema()) as c:
        docs = [Doc(id=f"d{i}",
                    fields={"body": " ".join(rng.choice(words, size=rng.integers(5, 20))),
                            "category": "bench"},
                    vectors={"vec": [0.0] * DIM}) for i in range(n)]
        insert_batched(c, docs)
        qs = [" ".join(rng.choice(words, size=rng.integers(1, 4))) for _ in range(nq)]
        t0 = time.perf_counter()
        for q in qs:
            c.query_text(TextQuery(field_name="body", text=q, topk=10))
        dt = time.perf_counter() - t0
    return R("fts_throughput_10k", dt, throughput=nq / dt,
             details={"n_docs": n, "n_queries": nq})


# ===================================================================
# 6. HYBRID SEARCH  (vec + FTS + RRF)
# ===================================================================

def bench_hybrid(n=1000):
    vecs = rvecs(n)
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    rng = np.random.default_rng(55)
    with tcoll("hybrid", fts_hnsw_schema()) as c:
        docs = [Doc(id=f"d{i}",
                    fields={"body": " ".join(rng.choice(words, size=rng.integers(3, 8))),
                            "category": "test"},
                    vectors={"vec": vecs[i].tolist()}) for i in range(n)]
        insert_batched(c, docs)
        c.optimize()

        errs = []
        t0 = time.perf_counter()
        res = c.query(vectors=VectorQuery("vec", vector=vecs[0].tolist()),
                      text=TextQuery(field_name="body", text="alpha", topk=50),
                      topk=10, reranker=RrfReRanker(topn=10))
        dt = time.perf_counter() - t0
        if len(res) == 0: errs.append("0 results")
        if len(res) > 10: errs.append(f"topk=10 got {len(res)}")

        try:
            c.query(vectors=VectorQuery("vec", vector=vecs[0].tolist()),
                    text=TextQuery(field_name="body", text="alpha"), topk=10)
            errs.append("no reranker should raise")
        except ValueError:
            pass

        if errs:
            return R("hybrid", dt, error="; ".join(errs), passed=False)
    return R("hybrid", dt, details={"n_results": len(res), "n_docs": n})


# ===================================================================
# 7. WAL DURABILITY COMPARISON  (10K docs)
# ===================================================================

def bench_wal(n=10_000):
    vecs = rvecs(n)
    timings = {}
    for name, mode in [("NONE", WalDurability.NONE), ("PER_BATCH", WalDurability.PER_BATCH)]:
        with tcoll(f"wal_{name}", hnsw_schema(),
                   CollectionOption(enable_mmap=False, wal_durability=mode)) as c:
            docs = [Doc(id=f"d{i}", vectors={"vec": vecs[i].tolist()}) for i in range(n)]
            t0 = time.perf_counter()
            insert_batched(c, docs)
            timings[name] = time.perf_counter() - t0
    return R("wal_durability_10k", timings["PER_BATCH"],
             details={f"{k}_sec": f"{v:.3f}" for k, v in timings.items()})


# ===================================================================
# 8. HNSW OPTIMIZE MULTI-BATCH  (50K, many segments)
# ===================================================================

def bench_optimize_hnsw(n=50_000):
    vecs = rvecs(n)
    with tcoll("opt_hnsw", hnsw_schema()) as c:
        for s in range(0, n, 500):
            batch = [Doc(id=f"d{i}", vectors={"vec": vecs[i].tolist()})
                     for i in range(s, min(s + 500, n))]
            c.insert(batch)

        t0 = time.perf_counter()
        c.optimize(OptimizeOption(concurrency=0, parallel_tasks=0))
        dt = time.perf_counter() - t0

        res = c.query(VectorQuery("vec", vector=vecs[0].tolist()), topk=5)
        if len(res) == 0:
            return R("optimize_hnsw_50k", dt, error="0 results after optimize", passed=False)
        if c.stats.doc_count != n:
            return R("optimize_hnsw_50k", dt, error=f"count {c.stats.doc_count}!={n}", passed=False)
    return R("optimize_hnsw_50k", dt, details={"n_docs": n, "segments_before": n // 500})


# ===================================================================
# 9. CANCEL TOKEN
# ===================================================================

def bench_cancel():
    errs = []
    tk = CancelToken()
    if tk.is_cancelled: errs.append("fresh is cancelled")
    tk.cancel()
    if not tk.is_cancelled: errs.append("not cancelled after cancel()")
    tk.cancel()
    if not tk.is_cancelled: errs.append("idempotent fail")
    if errs:
        return R("cancel_token", error="; ".join(errs), passed=False)
    return R("cancel_token")


# ===================================================================
# 10. AUTOOPTIMIZER (HNSW)
# ===================================================================

def bench_auto_opt():
    n = 1000
    vecs = rvecs(n)
    opt = CollectionOption(enable_mmap=False,
                           auto_optimize_enabled=True,
                           auto_optimize_interval_seconds=1,
                           auto_optimize_max_segments=3,
                           auto_optimize_cooldown_seconds=1)
    with tcoll("auto_opt", hnsw_schema(), opt) as c:
        for s in range(0, n, 50):
            c.insert([Doc(id=f"d{i}", vectors={"vec": vecs[i].tolist()})
                      for i in range(s, min(s + 50, n))])
        time.sleep(4)
        res = c.query(VectorQuery("vec", vector=vecs[0].tolist()), topk=5)
        if len(res) == 0:
            return R("auto_optimizer", error="0 results", passed=False)
        if c.stats.doc_count != n:
            return R("auto_optimizer", error=f"count {c.stats.doc_count}!={n}", passed=False)
    return R("auto_optimizer", details={"docs_intact": n})


# ===================================================================
# 11. MAX TOPK ENFORCEMENT
# ===================================================================

def bench_topk():
    vecs = rvecs(200)
    with tcoll("topk", hnsw_schema()) as c:
        c.insert([Doc(id=f"d{i}", vectors={"vec": vecs[i].tolist()}) for i in range(200)])
        c.optimize()
        errs = []

        for k in [1, 10, 100, 200]:
            r = c.query(VectorQuery("vec", vector=vecs[0].tolist()), topk=k)
            if len(r) != min(k, 200):
                errs.append(f"topk={k}: got {len(r)}")

        try:
            c.query(VectorQuery("vec", vector=vecs[0].tolist()), topk=MAX_TOPK + 1)
            errs.append(f"topk={MAX_TOPK + 1} should raise")
        except Exception:
            pass

        if errs:
            return R("max_topk", error="; ".join(errs), passed=False)
    return R("max_topk")


# ===================================================================
# 12. UPSERT & DELETE
# ===================================================================

def bench_upsert_delete():
    vecs = rvecs(20)
    with tcoll("ud", hnsw_schema()) as c:
        errs = []
        c.insert([Doc(id=f"d{i}", vectors={"vec": vecs[i].tolist()}) for i in range(10)])
        new_vec = [1.0] + [0.0] * (DIM - 1)
        c.upsert(Doc(id="d0", vectors={"vec": new_vec}))
        r = c.query(VectorQuery("vec", vector=new_vec), topk=1)
        if not r or r[0].id != "d0":
            errs.append(f"upsert: top is {r[0].id if r else 'empty'}")
        c.delete("d0")
        r = c.query(VectorQuery("vec", vector=new_vec), topk=10)
        if "d0" in {d.id for d in r}:
            errs.append("d0 still present after delete")
        if c.stats.doc_count != 9:
            errs.append(f"count {c.stats.doc_count}!=9")
        if errs:
            return R("upsert_delete", error="; ".join(errs), passed=False)
    return R("upsert_delete")


# ===================================================================
# 13. CONCURRENT R/W  (5K writes + 200 reads, 4 shards)
# ===================================================================

def bench_concurrent(n=5000):
    vecs = rvecs(n + 200)
    opt = CollectionOption(write_shards=4, enable_mmap=False)
    with tcoll("conc", hnsw_schema(), opt) as c:
        c.insert([Doc(id=f"d{i}", vectors={"vec": vecs[i].tolist()}) for i in range(200)])
        errs, wc, rc = [], [0], [0]

        def writer():
            for i in range(200, 200 + n):
                try:
                    c.insert(Doc(id=f"d{i}", vectors={"vec": vecs[i].tolist()}))
                    wc[0] += 1
                except Exception as e:
                    errs.append(f"w d{i}: {e}")

        def reader():
            for _ in range(200):
                try:
                    c.query(VectorQuery("vec", vector=vecs[0].tolist()), topk=5)
                    rc[0] += 1
                except Exception as e:
                    errs.append(f"r: {e}")

        t0 = time.perf_counter()
        ts = [threading.Thread(target=writer),
              threading.Thread(target=reader), threading.Thread(target=reader)]
        for t in ts: t.start()
        for t in ts: t.join()
        dt = time.perf_counter() - t0
        if errs:
            return R("concurrent_rw_5k", dt, error=f"{len(errs)} errs: {errs[:3]}", passed=False)
    return R("concurrent_rw_5k", dt, details={"writes": wc[0], "reads": rc[0]})


# ===================================================================
# 14. EDGE CASES
# ===================================================================

def bench_edge():
    errs = []
    with tcoll("e1", hnsw_schema()) as c:
        r = c.query(VectorQuery("vec", vector=[0.0] * DIM), topk=10)
        if len(r) != 0: errs.append(f"empty coll: {len(r)}")

    with tcoll("e2", hnsw_schema()) as c:
        c.insert(Doc(id="only", vectors={"vec": [1.0] + [0.0] * (DIM - 1)}))
        r = c.query(VectorQuery("vec", vector=[1.0] + [0.0] * (DIM - 1)), topk=10)
        if len(r) != 1 or r[0].id != "only": errs.append(f"single: {[d.id for d in r]}")

    with tcoll("e3", hnsw_schema()) as c:
        c.insert(Doc(id="dup", vectors={"vec": [0.0] * DIM}))
        s = c.insert(Doc(id="dup", vectors={"vec": [0.0] * DIM}))
        if s.ok(): errs.append("dup insert should fail")

    o = CollectionOption()
    if o.write_shards != 1: errs.append(f"default shards={o.write_shards}")
    if o.auto_optimize_enabled: errs.append("auto_opt on by default")

    oo = OptimizeOption()
    if oo.concurrency != 0 or oo.parallel_tasks != 0 or oo.memory_budget_bytes != 0:
        errs.append("OptimizeOption defaults wrong")

    fts = FtsIndexParam()
    if fts.tokenizer != "default" or abs(fts.k1 - 1.2) > 0.01 or abs(fts.b - 0.75) > 0.01:
        errs.append("FtsIndexParam defaults wrong")

    try:
        TextQuery(field_name="", text="hello")._validate()
        errs.append("empty field should fail")
    except ValueError: pass
    try:
        TextQuery(field_name="f", text="hello", topk=0)._validate()
        errs.append("topk=0 should fail")
    except ValueError: pass

    if errs:
        return R("edge_cases", error="; ".join(errs), passed=False)
    return R("edge_cases")


# ===================================================================
# 15. SHARD COMPARISON
# ===================================================================

def bench_shard_cmp(n=10_000):
    vecs = rvecs(n)
    timings = {}
    for shards in [1, 4]:
        with tcoll(f"sc_{shards}", hnsw_schema(),
                   CollectionOption(write_shards=shards, enable_mmap=False,
                                    wal_durability=WalDurability.NONE)) as c:
            docs = [Doc(id=f"d{i}", vectors={"vec": vecs[i].tolist()}) for i in range(n)]
            t0 = time.perf_counter()
            insert_batched(c, docs)
            timings[f"{shards}_shard"] = time.perf_counter() - t0
    return R("shard_comparison_10k", details={k: f"{v:.3f}s" for k, v in timings.items()})


# ===================================================================
# 16. FTS EDGE CASES
# ===================================================================

def bench_fts_edge():
    errs = []
    with tcoll("fts_e", fts_hnsw_schema()) as c:
        c.insert([
            Doc(id="d0", fields={"body": "hello world", "category": "a"}, vectors={"vec": [0.0] * DIM}),
            Doc(id="d1", fields={"body": "foo bar baz", "category": "b"}, vectors={"vec": [0.0] * DIM}),
        ])
        r = c.query_text(TextQuery(field_name="body", text="zzzz", topk=10))
        if len(r) != 0: errs.append(f"no-match: {len(r)}")
        r = c.query_text(TextQuery(field_name="body", text="hello", topk=1))
        if len(r) != 1: errs.append(f"topk=1: {len(r)}")
    if errs:
        return R("fts_edge", error="; ".join(errs), passed=False)
    return R("fts_edge")


# ===================================================================
# RUNNER
# ===================================================================

ALL = [
    ("Insert 50K (single-thread)", bench_insert),
    ("Sharded Insert 50K (4t x 4s)", bench_sharded),
    ("HNSW Optimize + Recall Sweep 50K (topk 10→20K)", bench_recall_sweep),
    ("FTS Correctness", bench_fts_correct),
    ("FTS Throughput 10K (500q)", bench_fts_throughput),
    ("Hybrid Search (vec+FTS+RRF)", bench_hybrid),
    ("WAL Durability 10K", bench_wal),
    ("HNSW Optimize Multi-Batch 50K", bench_optimize_hnsw),
    ("CancelToken", bench_cancel),
    ("AutoOptimizer (HNSW)", bench_auto_opt),
    ("Max TopK Enforcement", bench_topk),
    ("Upsert & Delete", bench_upsert_delete),
    ("Concurrent R/W 5K (4 shards)", bench_concurrent),
    ("Edge Cases", bench_edge),
    ("Shard 1 vs 4 (10K)", bench_shard_cmp),
    ("FTS Edge Cases", bench_fts_edge),
]


def run_all():
    ensure_init()
    print("=" * 76)
    print("ZVEC LARGE-SCALE BENCHMARK SUITE")
    print("=" * 76)
    print()

    results = []
    for label, fn in ALL:
        print(f"--- {label} ---", flush=True)
        try:
            r = fn()
        except Exception as e:
            r = R(label, error=str(e), passed=False)
        results.append(r)
        print(fmt(r), flush=True)
        print(flush=True)

    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)
    p = sum(1 for r in results if r.passed)
    f = sum(1 for r in results if not r.passed)
    print(f"  Passed: {p}/{len(results)}")
    print(f"  Failed: {f}/{len(results)}")
    if f:
        print("\nFailed:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
    print(flush=True)
    return 0 if f == 0 else 1


if __name__ == "__main__":
    code = run_all()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
else:
    import pytest

    @pytest.fixture(scope="session", autouse=True)
    def _init():
        ensure_init()

    @pytest.mark.parametrize("fn", [fn for _, fn in ALL], ids=[l for l, _ in ALL])
    def test_bench(fn):
        r = fn()
        print(f"\n{fmt(r)}")
        assert r.passed, r.error
