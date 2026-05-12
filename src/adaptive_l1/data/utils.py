from pathlib import Path
import random

import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_data_split(data_dir: str | Path, split_dir: str | Path):
    """Create split of the data for training, validation and testing.

    Args:
        data_dir: directory containing the data files (e.g. .h5 files of fastMRI)
        split_dir: directory where the split files should be saved
    """

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.h5"))
    rng = random.Random(42)
    rng.shuffle(files)

    def write_split(path, files):
        with open(path, "w") as f:
            for file in files:
                f.write(file.name + "\n")

    n_files = len(files)
    training_files = files[: int(0.8 * n_files)]
    validation_files = files[int(0.8 * n_files) : int(0.9 * n_files)]
    test_files = files[int(0.9 * n_files) :]

    write_split(Path(split_dir) / "fastmri_training.txt", training_files)
    write_split(Path(split_dir) / "fastmri_validation.txt", validation_files)
    write_split(Path(split_dir) / "fastmri_test.txt", test_files)


def read_split_file(data_dir: str | Path, split_file: str | Path) -> list[Path]:
    """Read split file and return list of files.

    Args:
        data_dir: directory containing the data files (e.g. .h5 files of fastMRI)
        split_file: split file
    """
    data_dir = Path(data_dir)
    with open(split_file) as f:
        return [data_dir / line.strip() for line in f if line.strip()]
