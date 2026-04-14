# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import pytest
import zvec
from zvec import (
    Collection,
    CollectionOption,
    CollectionSchema,
    DataType,
    Doc,
    FieldSchema,
    FlatIndexParam,
    FtsIndexParam,
    IndexType,
    MatchOp,
    MetricType,
    RrfReRanker,
    TextQuery,
    VectorQuery,
    VectorSchema,
)


# =================== FtsIndexParam unit tests ===================


class TestFtsIndexParam:
    def test_defaults(self):
        param = FtsIndexParam()
        assert param.tokenizer == "default"
        assert param.k1 == pytest.approx(1.2)
        assert param.b == pytest.approx(0.75)
        assert param.type == IndexType.FTS

    def test_custom_values(self):
        param = FtsIndexParam(tokenizer="default", k1=2.0, b=0.5)
        assert param.tokenizer == "default"
        assert param.k1 == pytest.approx(2.0)
        assert param.b == pytest.approx(0.5)

    def test_to_dict(self):
        d = FtsIndexParam(tokenizer="default", k1=1.5, b=0.4).to_dict()
        assert d["tokenizer"] == "default"
        assert d["k1"] == pytest.approx(1.5)
        assert d["b"] == pytest.approx(0.4)

    def test_pickle_round_trip(self):
        import pickle

        original = FtsIndexParam(tokenizer="default", k1=1.7, b=0.6)
        restored = pickle.loads(pickle.dumps(original))
        assert restored.tokenizer == "default"
        assert restored.k1 == pytest.approx(1.7)
        assert restored.b == pytest.approx(0.6)

    def test_readonly_properties(self):
        param = FtsIndexParam()
        with pytest.raises(AttributeError):
            param.tokenizer = "other"


# =================== TextQuery unit tests ===================


class TestTextQuery:
    def test_defaults(self):
        q = TextQuery(field_name="body", text="hello")
        assert q.field_name == "body"
        assert q.text == "hello"
        assert q.topk == 10
        assert q.op == MatchOp.OR
        assert q.output_fields is None

    def test_custom_values(self):
        q = TextQuery(
            field_name="body",
            text="hello world",
            topk=5,
            op=MatchOp.AND,
            output_fields=["title"],
        )
        assert q.topk == 5
        assert q.op == MatchOp.AND
        assert q.output_fields == ["title"]

    def test_validate_rejects_empty_field(self):
        q = TextQuery(field_name="", text="hello")
        with pytest.raises(ValueError):
            q._validate()

    def test_validate_rejects_zero_topk(self):
        q = TextQuery(field_name="body", text="hello", topk=0)
        with pytest.raises(ValueError):
            q._validate()


# =================== End-to-end collection tests ===================


@pytest.fixture
def fts_collection(tmp_path_factory) -> Collection:
    schema = CollectionSchema(
        name="fts_collection",
        fields=[
            FieldSchema(
                "body",
                DataType.STRING,
                nullable=True,
                index_param=FtsIndexParam(tokenizer="default"),
            ),
        ],
        vectors=[
            # Collections require >= 1 vector field; tiny dummy.
            VectorSchema(
                "vec",
                DataType.VECTOR_FP32,
                dimension=2,
                index_param=FlatIndexParam(metric_type=MetricType.IP),
            ),
        ],
    )
    temp_dir = tmp_path_factory.mktemp("zvec_fts")
    coll = zvec.create_and_open(
        path=str(temp_dir / "fts"),
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=False),
    )
    try:
        yield coll
    finally:
        try:
            coll.destroy()
        except Exception:
            pass


def _make_doc(pk: str, body: str) -> Doc:
    return Doc(
        id=pk,
        fields={"body": body},
        vectors={"vec": [0.0, 0.0]},
    )


class TestFtsCollectionQuery:
    def test_schema_exposes_fts_field(self, fts_collection: Collection):
        body = fts_collection.schema.field("body")
        assert body is not None
        assert body.data_type == DataType.STRING
        assert body.index_param is not None
        assert body.index_param.type == IndexType.FTS

    def test_basic_query_returns_matching_docs(
        self, fts_collection: Collection
    ):
        results = fts_collection.insert(
            [
                _make_doc("a", "the quick brown fox jumps over the lazy dog"),
                _make_doc("b", "totally unrelated content here"),
                _make_doc("c", "a fox and a hound"),
            ]
        )
        assert all(s.ok() for s in results)

        out = fts_collection.query_text(
            TextQuery(field_name="body", text="fox", topk=10)
        )
        pks = sorted(d.id for d in out)
        assert pks == ["a", "c"]

    def test_scores_are_descending_and_attached(
        self, fts_collection: Collection
    ):
        results = fts_collection.insert(
            [
                _make_doc("many", "ranking ranking ranking is great"),
                _make_doc("once", "a single mention of ranking"),
            ]
        )
        assert all(s.ok() for s in results)

        out = fts_collection.query_text(
            TextQuery(field_name="body", text="ranking", topk=10)
        )
        assert len(out) == 2
        assert out[0].id == "many"
        assert out[1].id == "once"
        assert out[0].score > out[1].score
        assert out[0].score > 0.0

    def test_and_mode_requires_all_terms(self, fts_collection: Collection):
        fts_collection.insert(
            [
                _make_doc("both", "the quick brown fox"),
                _make_doc("only_quick", "be quick about it"),
                _make_doc("only_fox", "a fox is a fox"),
            ]
        )

        out = fts_collection.query_text(
            TextQuery(
                field_name="body",
                text="quick fox",
                topk=10,
                op=MatchOp.AND,
            )
        )
        assert [d.id for d in out] == ["both"]

    def test_topk_limits_results(self, fts_collection: Collection):
        fts_collection.insert(
            [
                _make_doc(f"d_{i}", "common topic content")
                for i in range(5)
            ]
        )
        out = fts_collection.query_text(
            TextQuery(field_name="body", text="common", topk=3)
        )
        assert len(out) == 3

    def test_deleted_docs_excluded(self, fts_collection: Collection):
        fts_collection.insert(
            [
                _make_doc("alive", "alpha bravo"),
                _make_doc("doomed", "alpha charlie"),
            ]
        )
        del_results = fts_collection.delete(["doomed"])
        assert all(s.ok() for s in del_results)

        out = fts_collection.query_text(
            TextQuery(field_name="body", text="alpha", topk=10)
        )
        assert [d.id for d in out] == ["alive"]

    def test_unknown_field_raises(self, fts_collection: Collection):
        with pytest.raises(Exception):
            fts_collection.query_text(
                TextQuery(field_name="no_such_field", text="x", topk=10)
            )

    def test_output_fields_narrows_doc(self, fts_collection: Collection):
        fts_collection.insert([_make_doc("a", "alpha")])
        out = fts_collection.query_text(
            TextQuery(
                field_name="body",
                text="alpha",
                topk=10,
                output_fields=[],  # explicit empty -> drop all fields
            )
        )
        assert len(out) == 1
        assert out[0].id == "a"
        # No scalar fields should be present.
        assert out[0].fields == {} or out[0].fields == {"body": None}


# =================== Hybrid query() tests ===================


class TestHybridQuery:
    def test_query_text_only_via_query(self, fts_collection: Collection):
        fts_collection.insert(
            [
                _make_doc("a", "the quick brown fox"),
                _make_doc("b", "totally unrelated"),
            ]
        )
        out = fts_collection.query(
            text=TextQuery(field_name="body", text="fox", topk=10)
        )
        assert [d.id for d in out] == ["a"]

    def test_hybrid_without_reranker_raises(self, fts_collection: Collection):
        fts_collection.insert([_make_doc("a", "alpha")])
        with pytest.raises(ValueError, match="reranker"):
            fts_collection.query(
                vectors=VectorQuery(field_name="vec", vector=[0.0, 0.0]),
                text=TextQuery(field_name="body", text="alpha"),
                topk=10,
            )

    def test_hybrid_with_rrf_returns_fused_list(
        self, fts_collection: Collection
    ):
        # Doc a: matches both vector (closest to query vec) and text ("alpha").
        # Doc b: text-only match.
        # Doc c: vector-only match (no "alpha" in body).
        # All vectors are within the same FlatIndexParam(IP) field.
        fts_collection.insert(
            [
                Doc(
                    id="a",
                    fields={"body": "alpha bravo"},
                    vectors={"vec": [1.0, 0.0]},
                ),
                Doc(
                    id="b",
                    fields={"body": "alpha charlie"},
                    vectors={"vec": [0.0, 1.0]},
                ),
                Doc(
                    id="c",
                    fields={"body": "delta echo"},
                    vectors={"vec": [0.9, 0.1]},
                ),
            ]
        )
        out = fts_collection.query(
            vectors=VectorQuery(field_name="vec", vector=[1.0, 0.0]),
            text=TextQuery(field_name="body", text="alpha", topk=10),
            topk=5,
            reranker=RrfReRanker(topn=5),
        )
        ids = {d.id for d in out}
        # Doc a appears in both rankings — should be present.
        # At least one of b (text-only) or c (vector-only) should also appear.
        assert "a" in ids
        assert ids - {"a"}, "Hybrid should pull in candidates beyond the overlap"

    def test_hybrid_topk_caps_final_results(
        self, fts_collection: Collection
    ):
        fts_collection.insert(
            [
                _make_doc(f"d_{i}", "common topic content")
                for i in range(5)
            ]
        )
        out = fts_collection.query(
            vectors=VectorQuery(field_name="vec", vector=[0.0, 0.0]),
            text=TextQuery(field_name="body", text="common", topk=10),
            topk=2,
            reranker=RrfReRanker(topn=10),
        )
        assert len(out) == 2
