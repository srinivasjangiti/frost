"""
demo_catalog.py
Lightweight catalog manager and inference engine for the FROST Recruiter Demo.
Provides:
1. Fast bootstrapping of a compact, 1,000-movie real MovieLens catalog (<150 KB CSV).
2. Cached 384-d FAISS index and embeddings for instant startup on free cloud tiers.
3. Modular inference wrapper calling the repository's native retrieval and reranking pipeline.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import ssl
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMBED_MODEL
from src.llm_enrich import LLMEnricher, load_items_from_csv
from src.embeddings import build_embeddings, load_embeddings
from src.vector_index import build_faiss_index, load_faiss_index
from src.candidate_retrieval import get_candidates_for_user
from src.rerank_llm import CrossReranker
from src.rerank_two_head import combine_relevance_novelty, novelty_from_pop_rank
from src.vark_simulator import (
    build_user_profile_from_minimal,
    build_text_profile_from_viewed_items,
    simulate_vark_quiz_responses,
)
from src.metrics import intra_list_diversity

logger = logging.getLogger("frost.demo")

DEMO_DIR = PROJECT_ROOT / "data" / "demo"
DEMO_ITEMS_CSV = DEMO_DIR / "items_demo.csv"
DEMO_EMB_NPY = DEMO_DIR / "item_embeddings_demo.npy"
DEMO_ID2IDX_JSON = DEMO_DIR / "id2idx_demo.json"
DEMO_FAISS_INDEX = DEMO_DIR / "items_demo.faiss"

# Fallback curated movie catalog in case network is completely blocked
CURATED_FALLBACK_MOVIES = [
    {"item_id": "1", "title": "Toy Story (1995)", "genres": "Adventure|Animation|Children|Comedy|Fantasy", "pop": 215},
    {"item_id": "2", "title": "Jumanji (1995)", "genres": "Adventure|Children|Fantasy", "pop": 110},
    {"item_id": "47", "title": "Seven (a.k.a. Se7en) (1995)", "genres": "Mystery|Thriller", "pop": 203},
    {"item_id": "50", "title": "Usual Suspects, The (1995)", "genres": "Crime|Mystery|Thriller", "pop": 204},
    {"item_id": "110", "title": "Braveheart (1995)", "genres": "Action|Drama|War", "pop": 237},
    {"item_id": "150", "title": "Apollo 13 (1995)", "genres": "Adventure|Drama|IMAX", "pop": 201},
    {"item_id": "260", "title": "Star Wars: Episode IV - A New Hope (1977)", "genres": "Action|Adventure|Sci-Fi", "pop": 251},
    {"item_id": "296", "title": "Pulp Fiction (1994)", "genres": "Comedy|Crime|Drama|Thriller", "pop": 307},
    {"item_id": "318", "title": "Shawshank Redemption, The (1994)", "genres": "Crime|Drama", "pop": 317},
    {"item_id": "356", "title": "Forrest Gump (1994)", "genres": "Comedy|Drama|Romance|War", "pop": 329},
    {"item_id": "480", "title": "Jurassic Park (1993)", "genres": "Action|Adventure|Sci-Fi|Thriller", "pop": 238},
    {"item_id": "527", "title": "Schindler's List (1993)", "genres": "Drama|War", "pop": 220},
    {"item_id": "589", "title": "Terminator 2: Judgment Day (1991)", "genres": "Action|Sci-Fi", "pop": 224},
    {"item_id": "593", "title": "Silence of the Lambs, The (1991)", "genres": "Crime|Horror|Thriller", "pop": 279},
    {"item_id": "858", "title": "Godfather, The (1972)", "genres": "Crime|Drama", "pop": 192},
    {"item_id": "1196", "title": "Star Wars: Episode V - The Empire Strikes Back (1980)", "genres": "Action|Adventure|Sci-Fi", "pop": 211},
    {"item_id": "1198", "title": "Raiders of the Lost Ark (Indiana Jones) (1981)", "genres": "Action|Adventure", "pop": 200},
    {"item_id": "1210", "title": "Star Wars: Episode VI - Return of the Jedi (1983)", "genres": "Action|Adventure|Sci-Fi", "pop": 196},
    {"item_id": "2571", "title": "Matrix, The (1999)", "genres": "Action|Sci-Fi|Thriller", "pop": 278},
    {"item_id": "2959", "title": "Fight Club (1999)", "genres": "Action|Crime|Drama|Thriller", "pop": 218},
    {"item_id": "4993", "title": "Lord of the Rings: The Fellowship of the Ring, The (2001)", "genres": "Adventure|Fantasy", "pop": 261},
    {"item_id": "5952", "title": "Lord of the Rings: The Two Towers, The (2002)", "genres": "Adventure|Fantasy", "pop": 240},
    {"item_id": "7153", "title": "Lord of the Rings: The Return of the King, The (2003)", "genres": "Action|Adventure|Drama|Fantasy", "pop": 250},
    {"item_id": "58559", "title": "Dark Knight, The (2008)", "genres": "Action|Crime|Drama|IMAX", "pop": 240},
    {"item_id": "79132", "title": "Inception (2010)", "genres": "Action|Crime|Drama|Mystery|Sci-Fi|Thriller|IMAX", "pop": 230},
    {"item_id": "109487", "title": "Interstellar (2014)", "genres": "Sci-Fi|IMAX", "pop": 180},
    {"item_id": "112556", "title": "Gone Girl (2014)", "genres": "Drama|Mystery|Thriller", "pop": 120},
    {"item_id": "122904", "title": "Deadpool (2016)", "genres": "Action|Adventure|Comedy|Sci-Fi", "pop": 140},
    {"item_id": "134130", "title": "Martian, The (2015)", "genres": "Adventure|Drama|Sci-Fi", "pop": 150},
    {"item_id": "168252", "title": "Logan (2017)", "genres": "Action|Sci-Fi", "pop": 115},
]


def ensure_demo_catalog(max_movies: int = 1200) -> Path:
    """
    Downloads MovieLens-latest-small (978 KB) if necessary, extracts the top
    `max_movies` most popular movies with real genre and popularity tags,
    and writes them to `data/demo/items_demo.csv`.
    """
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    if DEMO_ITEMS_CSV.exists() and DEMO_ITEMS_CSV.stat().st_size > 1000:
        return DEMO_ITEMS_CSV

    print("[FROST Demo] Preparing compact MovieLens catalog for demo...")
    movies_data: Dict[str, Dict[str, Any]] = {}
    pop_counts: Dict[str, int] = {}

    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    download_success = False

    try:
        # GroupLens SSL cert often fails strict verification; use unverified context
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            content = resp.read()
        z = zipfile.ZipFile(io.BytesIO(content))

        # Parse ratings for popularity
        if "ml-latest-small/ratings.csv" in z.namelist():
            with z.open("ml-latest-small/ratings.csv") as rf:
                r_reader = csv.DictReader(io.TextIOWrapper(rf, encoding="utf-8"))
                for row in r_reader:
                    mid = str(row["movieId"])
                    pop_counts[mid] = pop_counts.get(mid, 0) + 1

        # Parse movies
        if "ml-latest-small/movies.csv" in z.namelist():
            with z.open("ml-latest-small/movies.csv") as mf:
                m_reader = csv.DictReader(io.TextIOWrapper(mf, encoding="utf-8"))
                for row in m_reader:
                    mid = str(row["movieId"])
                    movies_data[mid] = {
                        "item_id": mid,
                        "title": row["title"],
                        "genres": row.get("genres", ""),
                        "pop": pop_counts.get(mid, 1),
                    }
        download_success = True
        print(f"[FROST Demo] Downloaded and parsed {len(movies_data)} MovieLens items.")
    except Exception as e:
        print(f"[FROST Demo] Note: Could not download GroupLens zip ({e}). Using curated catalog fallback.")

    if not download_success or len(movies_data) < 20:
        for m in CURATED_FALLBACK_MOVIES:
            movies_data[m["item_id"]] = dict(m)

    # Sort by popularity descending and keep top max_movies
    sorted_items = sorted(
        movies_data.values(),
        key=lambda x: int(x.get("pop", 0)),
        reverse=True,
    )[:max_movies]

    # Write to DEMO_ITEMS_CSV
    with open(DEMO_ITEMS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "title", "genres", "pop"])
        writer.writeheader()
        for item in sorted_items:
            writer.writerow(item)

    print(f"[FROST Demo] Wrote {len(sorted_items)} items to {DEMO_ITEMS_CSV}")
    return DEMO_ITEMS_CSV


def load_or_build_demo_index() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray, Dict[str, int], Any]:
    """
    Loads or creates FAISS index, dense embeddings, and metadata for the demo catalog.
    Returns (raw_items, enriched_items, embeddings, id2idx, faiss_index).
    """
    ensure_demo_catalog()
    items = load_items_from_csv(DEMO_ITEMS_CSV)

    enr = LLMEnricher(backend="heuristic")
    enriched = enr.enrich_items_list(items)

    emb, id2idx = build_embeddings(
        enriched,
        out_npy=DEMO_EMB_NPY,
        map_path=DEMO_ID2IDX_JSON,
    )
    index = build_faiss_index(emb, index_path=DEMO_FAISS_INDEX)
    return items, enriched, emb, id2idx, index


class FROSTDemoEngine:
    """
    Inference orchestrator for the Streamlit UI.
    Wraps candidate retrieval, cross-encoder reranking, and two-head multi-objective Pareto balancing.
    """

    def __init__(self):
        self.items, self.enriched, self.emb, self.id2idx, self.index = load_or_build_demo_index()
        self.item_by_id = {str(it["item_id"]): it for it in self.enriched}
        self.catalog_size = len(self.enriched)

        # Precalculate popularity ranking for novelty scoring
        item_pop_counts = {str(it["item_id"]): float(it.get("pop", 0) or 0) for it in self.enriched}
        sorted_by_pop = sorted(item_pop_counts.items(), key=lambda x: x[1], reverse=True)
        self.item_pop_rank = {iid: rank + 1 for rank, (iid, _) in enumerate(sorted_by_pop)}
        self.novelty_scores = novelty_from_pop_rank(self.item_pop_rank, catalog_size=self.catalog_size)

        # Lazy load Cross-Encoder only when reranking is triggered
        self._reranker: Optional[CrossReranker] = None

    def get_reranker(self) -> CrossReranker:
        if self._reranker is None:
            self._reranker = CrossReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                items_meta=self.enriched,
                device="cpu",
                item_embeddings=self.emb,
                id2idx=self.id2idx,
            )
        return self._reranker

    def recommend(
        self,
        scenario: str = "cold_start",  # "cold_start", "few_shot_1", "few_shot_5"
        viewed_item_ids: Optional[List[str]] = None,
        retrieval_mode: str = "hybrid",  # "hybrid", "ann", "bm25", "popularity", "embedding_cosine"
        candidate_pool_size: int = 100,
        use_reranker: bool = True,
        novelty_weight_alpha: float = 0.7,  # 1.0 = Pure Relevance, 0.0 = Pure Novelty
        top_k: int = 10,
        user_id: str = "demo_recruiter",
        time_of_day: str = "evening",
        device: str = "desktop",
        filter_already_seen: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference and returns recommendations with explanation and metrics.
        """
        t0 = time.time()
        viewed_ids = [str(x) for x in (viewed_item_ids or [])]

        # 1. User Profile Generation
        info = {
            "user_id": user_id,
            "goal": "",
            "time_of_day": time_of_day,
            "session_len": 15,
            "device": device,
            "viewed_item_ids": viewed_ids,
            "items_meta": self.enriched,
        }
        profile = build_user_profile_from_minimal(info, prior_mode="prior_plus_context")

        text_profile = profile.get("text_profile", "")
        vark_scores = profile.get("preference_prior", {}).get("vark_scores", {})
        dominant_vark = profile.get("preference_prior", {}).get("dominant", "visual")

        # 2. Stage 1: Candidate Retrieval
        candidates: List[str] = []
        retrieval_meta: Dict[str, Any] = {}

        if retrieval_mode == "popularity":
            sorted_items = sorted(self.enriched, key=lambda x: float(x.get("pop", 0) or 0), reverse=True)
            candidates = [str(it["item_id"]) for it in sorted_items[:candidate_pool_size]]
            retrieval_meta = {"retrieval_mode": "popularity", "pool_size": len(candidates)}
        elif retrieval_mode == "embedding_cosine":
            qtext = text_profile or f"user_vark:{dominant_vark}"
            from src.embeddings import get_embedder
            embedder = get_embedder(EMBED_MODEL)
            user_vec = embedder.encode([qtext], convert_to_numpy=True)[0]
            sims = []
            for it in self.enriched:
                iid = str(it["item_id"])
                idx = self.id2idx.get(iid)
                if idx is not None and idx < len(self.emb):
                    cos_sim = float(np.dot(user_vec, self.emb[idx]) / (np.linalg.norm(user_vec) * np.linalg.norm(self.emb[idx]) + 1e-9))
                    sims.append((iid, cos_sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            candidates = [iid for iid, _ in sims[:candidate_pool_size]]
            retrieval_meta = {"retrieval_mode": "embedding_cosine", "pool_size": len(candidates)}
        else:
            candidates, retrieval_meta = get_candidates_for_user(
                user_profile=profile,
                items_list=self.enriched,
                faiss_index=self.index,
                id2idx=self.id2idx,
                embeddings=self.emb,
                pool_size=candidate_pool_size,
                retrieval_mode=retrieval_mode,
            )

        # Filter already-seen items if requested
        if filter_already_seen and viewed_ids:
            viewed_set = set(viewed_ids)
            candidates = [c for c in candidates if c not in viewed_set]

        if not candidates:
            candidates = [str(it["item_id"]) for it in self.enriched[:top_k]]

        retrieval_latency_ms = (time.time() - t0) * 1000

        # 3. Stage 2 & 3: Neural Reranking and Multi-Objective Pareto Balancing
        t_rerank_start = time.time()
        scored_candidates: List[Dict[str, Any]] = []

        if use_reranker and retrieval_mode not in ("popularity", "random"):
            reranker = self.get_reranker()
            candidates_to_rerank = candidates[:min(len(candidates), candidate_pool_size)]
            reranked = reranker.rerank(profile, candidates_to_rerank, topk=len(candidates_to_rerank))

            # Combine Relevance with Novelty
            combined = combine_relevance_novelty(
                scored_list=reranked,
                item_novelty=self.novelty_scores,
                alpha=float(novelty_weight_alpha),
            )
            scored_candidates = combined[:top_k]
        else:
            # Baseline scoring without cross-encoder
            for rank_idx, iid in enumerate(candidates[:top_k]):
                novelty = self.novelty_scores.get(str(iid), 0.5)
                rel = max(0.0, 1.0 - (rank_idx / max(1, top_k)))
                final_score = float(novelty_weight_alpha * rel + (1.0 - novelty_weight_alpha) * novelty)
                scored_candidates.append({
                    "item_id": str(iid),
                    "score": final_score,
                    "relevance": rel,
                    "novelty": novelty,
                })

        rerank_latency_ms = (time.time() - t_rerank_start) * 1000
        total_latency_ms = (time.time() - t0) * 1000

        # 4. Assemble Final Results with Rich Metadata
        recommended_items = []
        rec_ids = []
        for rank, rec in enumerate(scored_candidates[:top_k], start=1):
            iid = str(rec.get("item_id"))
            rec_ids.append(iid)
            meta = self.item_by_id.get(iid, {})
            pop_rank = self.item_pop_rank.get(iid, len(self.item_pop_rank))
            novelty_val = rec.get("novelty", self.novelty_scores.get(iid, 0.5))
            rel_val = rec.get("relevance", rec.get("score", 0.0))

            # Diagnostic novelty badge
            if pop_rank < 50:
                tier = "🔥 Blockbuster"
            elif pop_rank < 250:
                tier = "⭐ Popular"
            else:
                tier = "💎 Discovery (Long-Tail)"

            recommended_items.append({
                "rank": rank,
                "item_id": iid,
                "title": meta.get("title", f"Movie #{iid}"),
                "genres": meta.get("genres", "General"),
                "score": float(rec.get("score", 0.0)),
                "relevance": float(rel_val),
                "novelty": float(novelty_val),
                "pop_rank": pop_rank,
                "tier": tier,
            })

        # 5. Compute Real Metrics
        emb_dict = {str(it["item_id"]): self.emb[self.id2idx[str(it["item_id"])]] for it in self.enriched if str(it["item_id"]) in self.id2idx}
        ild = intra_list_diversity([rec_ids], emb_dict) if len(rec_ids) >= 2 else 0.0
        avg_novelty = float(np.mean([r["novelty"] for r in recommended_items])) if recommended_items else 0.0

        return {
            "profile": {
                "text_profile": text_profile,
                "dominant_vark": dominant_vark,
                "vark_scores": vark_scores,
                "viewed_titles": [self.item_by_id.get(str(i), {}).get("title", f"Movie #{i}") for i in viewed_ids],
            },
            "recommendations": recommended_items,
            "metrics": {
                "ild": round(float(ild), 4),
                "avg_novelty": round(avg_novelty, 3),
                "total_latency_ms": round(total_latency_ms, 1),
                "retrieval_latency_ms": round(retrieval_latency_ms, 1),
                "pool_size": len(candidates),
                "retrieval_mode": retrieval_mode,
            },
        }


if __name__ == "__main__":
    print("Testing FROSTDemoEngine initialization...")
    engine = FROSTDemoEngine()
    print(f"Catalog size: {engine.catalog_size} movies.")
    print(f"FAISS index total vectors: {engine.index.ntotal}")

    print("\nTesting inference on 1-shot (Inception 79132)...")
    res = engine.recommend(
        scenario="few_shot_1",
        viewed_item_ids=["79132"],
        retrieval_mode="hybrid",
        candidate_pool_size=50,
        use_reranker=True,
        top_k=5,
    )
    print("Metrics:", res["metrics"])
    print("Top Recommendations:")
    for r in res["recommendations"]:
        print(f"  #{r['rank']}: {r['title']} ({r['genres']}) - Score: {r['score']:.3f} (Rel: {r['relevance']:.3f}, Nov: {r['novelty']:.3f})")
    print("\nSUCCESS: FROST pipeline end-to-end inference verified!")
