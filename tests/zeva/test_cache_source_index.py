from pathlib import Path


def test_cache_builders_bind_rows_to_returned_dataset_index():
    source = Path("scripts/build_zeva_robotwin_cache.py").read_text()
    assert 'source_index = int(sample.get("dataset_index", index))' in source
    assert '"window_index": source_index' in source
