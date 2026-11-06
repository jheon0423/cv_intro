import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


DOWNLOAD_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_data_source(data_path):
    location = Path(data_path).expanduser()
    if location.is_file():
        return location

    extracted = location / "cifar-10-batches-py"
    if (extracted / "data_batch_1").exists():
        return extracted
    if (location / "data_batch_1").exists():
        return location

    archive = location / "cifar-10-python.tar.gz"
    if not archive.exists():
        location.mkdir(parents=True, exist_ok=True)
        temporary_archive = location / "cifar-10-python.tar.gz.part"
        try:
            print(f"Downloading CIFAR-10 from {DOWNLOAD_URL}", flush=True)
            urllib.request.urlretrieve(DOWNLOAD_URL, temporary_archive)
            temporary_archive.replace(archive)
        except Exception as error:
            temporary_archive.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download CIFAR-10: {error}. Place cifar-10-python.tar.gz in {location} "
                "or pass --data to an existing archive or extracted dataset directory."
            ) from error
    return archive


def read_batch(source, filename):
    if source.is_dir():
        with (source / filename).open("rb") as file:
            record = pickle.load(file, encoding="bytes")
    else:
        with tarfile.open(source, "r:gz") as archive:
            member = archive.extractfile(f"cifar-10-batches-py/{filename}")
            if member is None:
                raise FileNotFoundError(f"Missing CIFAR-10 batch: {filename}")
            with member:
                record = pickle.load(member, encoding="bytes")
    return np.asarray(record[b"data"], dtype=np.uint8), np.asarray(record[b"labels"], dtype=np.int64)


def load_cifar10(data_path):
    source = get_data_source(data_path)
    train_batches = [read_batch(source, f"data_batch_{index}") for index in range(1, 6)]
    x_train = np.concatenate([batch[0] for batch in train_batches], axis=0)
    y_train = np.concatenate([batch[1] for batch in train_batches], axis=0)
    x_test, y_test = read_batch(source, "test_batch")
    return x_train, y_train, x_test, y_test


def select_subset(features, labels, sample_size, random_state):
    if sample_size == 0 or sample_size >= len(labels):
        indices = np.arange(len(labels))
    else:
        if sample_size < len(np.unique(labels)):
            raise ValueError("Sample size must include at least one sample per class")
        indices, _ = train_test_split(
            np.arange(len(labels)),
            train_size=sample_size,
            stratify=labels,
            random_state=random_state,
        )
    sampled_features = np.ascontiguousarray(features[indices], dtype=np.float32) / 255.0
    return sampled_features, labels[indices]
