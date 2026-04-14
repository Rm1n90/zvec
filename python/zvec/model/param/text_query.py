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

from dataclasses import dataclass
from typing import Optional

from _zvec.param import MatchOp

__all__ = ["TextQuery", "MatchOp"]


@dataclass(frozen=True)
class TextQuery:
    """Represents a full-text-search (BM25) query for an FTS-indexed STRING field.

    The text is tokenized at query time using the same tokenizer the field was
    indexed with. A document matches when the chosen ``op`` semantics hold over
    the resulting term set. Top-K results are ranked by descending BM25 score.

    Attributes:
        field_name (str): Name of the STRING field that has an FTS index.
        text (str): Query text.
        topk (int, optional): Number of results to return. Defaults to 10.
        op (MatchOp, optional): Combinator across query terms.

            - ``MatchOp.OR`` (default): any term matches; per-term BM25 scores summed.
            - ``MatchOp.AND``: doc must contain every term.

        output_fields (Optional[list[str]], optional): Fields to include in the
            returned ``Doc`` objects. ``None`` keeps all fields; an empty list
            drops every field (the doc still carries its id and score).

    Examples:
        >>> import zvec
        >>> q = zvec.TextQuery(
        ...     field_name="body",
        ...     text="quick brown fox",
        ...     topk=10,
        ...     op=zvec.MatchOp.OR,
        ... )
    """

    field_name: str
    text: str
    topk: int = 10
    op: MatchOp = MatchOp.OR
    output_fields: Optional[list[str]] = None

    def _validate(self) -> None:
        if not isinstance(self.field_name, str) or not self.field_name:
            raise ValueError("TextQuery: field_name must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("TextQuery: text must be a string")
        if not isinstance(self.topk, int) or self.topk <= 0:
            raise ValueError("TextQuery: topk must be a positive int")
        if not isinstance(self.op, MatchOp):
            raise TypeError("TextQuery: op must be a MatchOp value")
