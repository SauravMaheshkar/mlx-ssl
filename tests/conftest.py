import json
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from ml_collections.config_dict import ConfigDict


@pytest.fixture
def fetch_config(path_or_hf_repo: str, revision: str):
    path = Path(
        snapshot_download(
            repo_id=path_or_hf_repo,
            revision=revision,
            allow_patterns=[
                "*.json",
            ],
        )
    )

    config = json.loads(open(path / "config.json").read())

    return ConfigDict(config, allow_dotted_keys=True)
