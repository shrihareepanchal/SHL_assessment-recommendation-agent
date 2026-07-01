from __future__ import annotations

import pytest

from app.catalog.loader import CatalogLoadError, CatalogRepository


def test_loads_all_catalog_entries(catalog_path):
    repo = CatalogRepository(catalog_path)
    assert len(repo) > 300  # bundled catalog has 377 Individual Test Solutions
    assert len(repo.all) == len(repo)


def test_every_entry_has_name_and_url(catalog_path):
    repo = CatalogRepository(catalog_path)
    for assessment in repo.all:
        assert assessment.name
        assert assessment.url.startswith("https://www.shl.com/")


def test_is_valid_url_true_for_catalog_entry(catalog_path):
    repo = CatalogRepository(catalog_path)
    real_url = repo.all[0].url
    assert repo.is_valid_url(real_url) is True


def test_is_valid_url_false_for_fabricated_url(catalog_path):
    repo = CatalogRepository(catalog_path)
    assert repo.is_valid_url("https://www.shl.com/products/product-catalog/view/does-not-exist/") is False


def test_missing_file_raises_catalog_load_error(tmp_path):
    with pytest.raises(CatalogLoadError):
        CatalogRepository(tmp_path / "nonexistent.json")


def test_fuzzy_find_resolves_common_abbreviation(catalog_path):
    repo = CatalogRepository(catalog_path)
    matches = repo.fuzzy_find_by_name_fragment("OPQ32r")
    assert any("OPQ32r" in m.name for m in matches)
