#!/usr/bin/env python3
"""kg_search.py - graph-neighbor retrieval over the dormant KG (containers.jsonl).

v1 of the COLD read path's "semantic search": given a query entity, return its
nearest neighbors in the real typed-edge graph - entities that share edge-targets
(hubs) with the query, plus entities directly linked to/from it. No embeddings
needed: the edges ARE the semantics. v2 swaps in the semantic core's learned
embeddings (see cold-read-structured-knowledge) once it's trained on real data.

The hits this returns are what surface into the `structured_knowledge` region of
the active field, exactly as the runtime will do per input. populate_structured_
knowledge (cold_read.py) renders them into that region for the core to attend.
"""
from __future__ import annotations
import json
from collections import defaultdict

SKIP_EDGE = "described by"   # 70% noise relation; ignore (same as recall curriculum)


class KGSearch:
    def __init__(self, path: str):
        self.by_word: dict[str, dict] = {}            # word -> container
        self.edges: dict[str, list] = defaultdict(list)        # word -> [(etype, hub_norm)]
        self.hub_to_words: dict[str, set] = defaultdict(set)   # hub_norm -> {words linking to it}
        self.norm_to_words: dict[str, list] = defaultdict(list)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                w = o.get("word", "").strip()
                if not w:
                    continue
                self.by_word[w] = o
                for et, tg in o.get("edges", []):
                    if et == SKIP_EDGE:
                        continue
                    hub = self._norm(tg)
                    if not hub:
                        continue
                    self.edges[w].append((et, hub))
                    self.hub_to_words[hub].add(w)
        for w in self.by_word:
            self.norm_to_words[self._norm(w)].append(w)

    @staticmethod
    def _norm(s: str) -> str:
        return s.lower().strip()

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Neighbor containers most related to `query`, ranked deterministically.

        Pass k only for an explicit retrieval/display budget. By default the
        full ranked result set is returned.
        """
        qn = self._norm(query)
        qwords = self.norm_to_words.get(qn, [])
        qw = qwords[0] if qwords else None
        qhubs = {hub for _, hub in self.edges.get(qw, [])} if qw else set()

        scores: dict[str, int] = defaultdict(int)
        for h in qhubs:
            for w in self.hub_to_words.get(h, ()):        # siblings sharing a hub
                if w != qw:
                    scores[w] += 1
            for w in self.norm_to_words.get(h, ()):       # hub is itself an entity = direct link
                if w != qw:
                    scores[w] += 2
        for w in self.hub_to_words.get(qn, ()):           # reverse links (point AT query)
            if w != qw:
                scores[w] += 2

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if k is not None:
            ranked = ranked[:k]
        return [self.by_word[w] for w, _ in ranked]

    def render(self, container: dict, max_edges: int | None = None) -> str:
        """Render a hit as the text that goes into structured_knowledge:
        'word: etype target; etype target'.

        Pass max_edges only for an explicit display/curriculum budget. By
        default all semantic edges render.
        """
        w = container.get("word", "")
        es = [(et, tg) for et, tg in container.get("edges", []) if et != SKIP_EDGE]
        if max_edges is not None:
            es = es[:max_edges]
        body = "; ".join(f"{et} {tg}" for et, tg in es)
        return f"{w}: {body}" if body else w


def _selftest():
    import pathlib
    path = pathlib.Path(__file__).resolve().parent / "datasets/containers/containers.jsonl"
    kg = KGSearch(str(path))
    print(f"[kg] loaded {len(kg.by_word)} entities, {len(kg.hub_to_words)} hubs")
    for q in ("axon", "jeffrey", "powershell", "fastapi", ".cloudflared", "WMI"):
        hits = kg.search(q, k=5)
        print(f"\n[query] {q!r} -> {len(hits)} hits")
        for h in hits:
            print("   ", kg.render(h)[:110])


if __name__ == "__main__":
    _selftest()
