from dataclasses import dataclass

from app.backend.catalog import HuggingFaceCatalog, extract_quantization, parse_model_metadata


def test_extract_quantization() -> None:
    assert extract_quantization("model-Q4_K_M.gguf") == "Q4_K_M"
    assert extract_quantization("model.iq3_xxs.gguf") == "IQ3_XXS"
    assert extract_quantization("model.gguf") == "Unknown"


@dataclass
class FakeSibling:
    rfilename: str
    size: int


@dataclass
class FakeInfo:
    id: str = "org/model-GGUF"
    siblings: list[FakeSibling] | None = None
    downloads: int = 42
    likes: int = 7
    tags: list[str] | None = None
    card_data: dict | None = None
    config: dict | None = None


class FakeApi:
    def __init__(self, info: FakeInfo) -> None:
        self.info = info

    def model_info(self, repo_id: str, files_metadata: bool = False) -> FakeInfo:
        return self.info


def test_metadata_missing_fields_are_unknown() -> None:
    item = parse_model_metadata(FakeInfo(tags=[], card_data={}, config={}))
    assert item.license == "Unknown"
    assert item.architecture == "Unknown"
    assert item.context_length is None


def test_discovers_and_sums_split_gguf() -> None:
    info = FakeInfo(
        siblings=[
            FakeSibling("model-Q4_K_M-00001-of-00002.gguf", 10),
            FakeSibling("model-Q4_K_M-00002-of-00002.gguf", 15),
            FakeSibling("model-Q8_0.gguf", 40),
            FakeSibling("README.md", 100),
        ]
    )
    files = HuggingFaceCatalog(FakeApi(info)).discover_gguf(info.id)
    assert len(files) == 2
    split = next(item for item in files if item.quantization == "Q4_K_M")
    assert split.size_bytes == 25
    assert len(split.split_files) == 2
