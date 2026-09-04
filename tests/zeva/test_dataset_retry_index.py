from pathlib import Path


def test_robot_video_dataset_records_successful_source_index():
    source = Path("src/fastwam/datasets/lerobot/robot_video_dataset.py").read_text()
    assert '"dataset_index": int(sample.get("idx", sample_idx))' in source
