"""Bounded, auditable tools over the play's fictional evidence corpus.

The lexical retriever is deliberately local: every quotation can be traced back
to a UTF-8 file without embeddings, a vector service, or an extra API key.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


def _tokens(text: str) -> list[str]:
    # Prefix normalization helps Russian inflections without another dependency.
    return [
        word[:6] for word in re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е")) if len(word) > 2
    ]


class ToolBox:
    """Read-only archive tools plus a transparent preservation cost calculator."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Archive root must be a directory")
        self.documents: dict[str, dict] = {}
        for path in sorted(self.root.glob("*.md")):
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                continue
            match = re.match(r"(D\d+)_", path.name)
            if not match:
                continue
            doc_id = match.group(1)
            if doc_id in self.documents:
                raise ValueError(f"Duplicate document ID: {doc_id}")
            if path.stat().st_size > 100_000:
                raise ValueError(f"Archive document too large: {path.name}")
            self.documents[doc_id] = {
                "id": doc_id,
                "path": path.name,
                "file": path,
                "text": path.read_text(encoding="utf-8"),
            }
        if not self.documents:
            raise ValueError("Archive has no D<number>_*.md documents")

    def schemas(self) -> list[dict]:
        return [
            {
                "name": "search_files",
                "description": "Поиск свидетельств в вымышленном городском архиве. Вернёт ID источников для цитирования.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "read_file",
                "description": "Прочитать найденный архивный документ по ID D1–D5 или безопасному имени файла. Если указаны оба, они должны относиться к одному документу.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "path": {"type": "string"}},
                    "anyOf": [{"required": ["id"]}, {"required": ["path"]}],
                    "additionalProperties": False,
                },
            },
            {
                "name": "estimate_preservation_cost",
                "description": "Смета оцифровки: 40 страниц на одного волонтёра за одну восьмичасовую смену, остальные страницы оплачиваются по тарифу. Это упрощённая модель внутри сюжета.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pages": {"type": "integer", "minimum": 0, "maximum": 1000000},
                        "cost_per_page": {"type": "number", "minimum": 0, "maximum": 1000000},
                        "volunteers": {"type": "integer", "minimum": 0, "maximum": 10000},
                    },
                    "required": ["pages"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, arguments: dict) -> dict:
        """Return errors as data so the agent can revise an invalid request."""
        if not isinstance(arguments, dict):
            return {"error": "Tool arguments must be an object"}
        methods = {
            "search_files": self.search_files,
            "read_file": self.read_file,
            "estimate_preservation_cost": self.estimate_preservation_cost,
        }
        if not isinstance(name, str) or name not in methods:
            return {"error": f"Unknown tool: {str(name)[:80]}"}
        try:
            return methods[name](**arguments)
        except (TypeError, ValueError, OSError) as exc:
            return {"error": str(exc)[:300]}

    def search_files(self, query: str, limit: int = 3) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 400:
            raise ValueError("query must contain 1–400 characters")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5:
            raise ValueError("limit must be an integer from 1 to 5")
        words = set(_tokens(query))
        counts = {key: Counter(_tokens(doc["text"])) for key, doc in self.documents.items()}
        total = len(counts)
        hits = []
        for key, doc in self.documents.items():
            score = 0.0
            for word in words:
                frequency = counts[key][word]
                containing = sum(word in counter for counter in counts.values())
                if frequency:
                    score += math.log(1 + (total + 1) / (containing + 1)) * frequency / (frequency + 1.2)
            if score:
                # Whole documents fit the bounded context for this small corpus.
                hits.append(
                    {"id": key, "path": doc["path"], "text": doc["text"][:2400], "score": round(score, 4)}
                )
        hits.sort(key=lambda item: (-item["score"], item["id"]))
        return {"query": query, "hits": hits[:limit], "fictional": True}

    def read_file(self, path: str | None = None, id: str | None = None) -> dict:
        if path is None and id is None:
            raise ValueError("Provide id or path")
        if id is not None:
            if not isinstance(id, str) or id not in self.documents:
                raise ValueError("Unknown document ID")
            doc = self.documents[id]
            if path is not None and path != doc["path"]:
                raise ValueError("id and path must identify the same indexed document")
        else:
            if (
                not isinstance(path, str)
                or not path
                or any(char in path for char in ("/", "\\", ":"))
                or ".." in path
            ):
                raise ValueError("Only an indexed archive filename is allowed")
            doc = next((item for item in self.documents.values() if item["path"] == path), None)
            if doc is None:
                raise ValueError("File is outside the indexed archive")
        # Recheck before reading so a replaced symlink cannot escape the corpus.
        file = doc["file"]
        if file.is_symlink() or not file.resolve(strict=True).is_relative_to(self.root):
            raise ValueError("File is outside the indexed archive")
        if file.stat().st_size > 100_000:
            raise ValueError("Archive document too large")
        text = file.read_text(encoding="utf-8")
        return {
            "id": doc["id"],
            "path": doc["path"],
            "text": text[:6000],
            "truncated": len(text) > 6000,
            "fictional": True,
        }

    @staticmethod
    def estimate_preservation_cost(pages: int, cost_per_page: float = 2.5, volunteers: int = 0) -> dict:
        for value, label, upper in ((pages, "pages", 1_000_000), (volunteers, "volunteers", 10_000)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= upper:
                raise ValueError(f"{label} must be an integer from 0 to {upper}")
        if (
            isinstance(cost_per_page, bool)
            or not isinstance(cost_per_page, (float, int))
            or not math.isfinite(cost_per_page)
            or not 0 <= cost_per_page <= 1_000_000
        ):
            raise ValueError("cost_per_page must be a finite nonnegative number <= 1000000")
        paid_pages = max(0, pages - volunteers * 40)
        cost = (Decimal(str(cost_per_page)) * paid_pages).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return {
            "pages": pages,
            "volunteers": volunteers,
            "volunteer_pages": min(pages, volunteers * 40),
            "paid_pages": paid_pages,
            "cost_per_page": cost_per_page,
            "total_rub": float(cost),
            "assumption": "Одна смена 8 часов; 40 страниц на волонтёра. Модель сметы вымышленная.",
        }
