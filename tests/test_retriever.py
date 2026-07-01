from __future__ import annotations

import shutil

import pytest

from app.catalog.loader import CatalogRepository
from app.ranking.metadata_scorer import ConstraintProfile, score_metadata_fit
from app.retriever.embedder import get_embedder
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.keyword_search import KeywordIndex
from app.retriever.vector_store import VectorStore


@pytest.fixture(scope="module")
def catalog(catalog_path):
    return CatalogRepository(catalog_path)


@pytest.fixture(scope="module")
def retriever(catalog, tmp_path_factory):
    persist_dir = str(tmp_path_factory.mktemp("chroma_test"))
    embedder = get_embedder()
    vector_store = VectorStore(persist_dir, "test_collection", embedder)
    vector_store.rebuild(catalog.all)
    keyword_index = KeywordIndex(catalog.all)
    hybrid = HybridRetriever(
        catalog=catalog,
        vector_store=vector_store,
        keyword_index=keyword_index,
        top_k_semantic=25,
        top_k_keyword=25,
        semantic_weight=0.6,
        keyword_weight=0.4,
        metadata_weight=0.15,
    )
    yield hybrid
    shutil.rmtree(persist_dir, ignore_errors=True)


def test_keyword_search_finds_exact_product_name(catalog):
    index = KeywordIndex(catalog.all)
    hits = index.query("OPQ32r", top_k=5)
    assert hits, "expected at least one BM25 hit for an exact product code"
    top_id = hits[0][0]
    assert "OPQ32r" in catalog.get_by_id(top_id).name


def test_hybrid_retrieval_returns_relevant_java_assessment(retriever, catalog):
    results = retriever.retrieve("Java developer knowledge test", ConstraintProfile(), final_k=10)
    names = [r.assessment.name.lower() for r in results]
    assert any("java" in n for n in names)


def test_metadata_scorer_neutral_when_no_constraints(catalog):
    assessment = catalog.all[0]
    assert score_metadata_fit(assessment, ConstraintProfile()) == 0.0


def test_metadata_scorer_rewards_matching_test_type(catalog):
    personality = next(a for a in catalog.all if "P" in a.test_type_codes)
    constraints = ConstraintProfile(test_type_codes=["P"])
    assert score_metadata_fit(personality, constraints) > 0.0
