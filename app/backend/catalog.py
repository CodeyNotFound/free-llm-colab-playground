from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from huggingface_hub import HfApi, ModelCard

from .inference.models import GGUFFile, ModelMetadata

QUANT_PATTERN = re.compile(r"(?:^|[._-])(IQ\d(?:_[A-Z]+)?|Q\d(?:_[A-Z0-9]+(?:_[A-Z0-9]+)*)?)(?:[._-]|$)", re.I)
SPLIT_PATTERN = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


def extract_quantization(filename: str) -> str:
    match = QUANT_PATTERN.search(filename)
    return match.group(1).upper() if match else "Unknown"


def _card_description(repo_id: str) -> str:
    try:
        card = ModelCard.load(repo_id)
        text = card.text.strip()
        # First useful paragraph, never pretend it is structured metadata.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return next((p for p in paragraphs if not p.startswith(("---", "#", "<"))), "Unknown")[:800]
    except Exception:  # HF cards vary widely and failure should not break discovery.
        return "Unknown"


def parse_model_metadata(info: Any, *, include_description: bool = False) -> ModelMetadata:
    repo_id = getattr(info, "id", None) or getattr(info, "modelId", "Unknown")
    card_data = getattr(info, "card_data", None) or {}
    config = getattr(info, "config", None) or {}
    tags = tuple(getattr(info, "tags", None) or ())
    if not isinstance(card_data, dict):
        card_data = dict(card_data) if hasattr(card_data, "items") else {}
    if not isinstance(config, dict):
        config = {}
    architectures = config.get("architectures") or []
    architecture = architectures[0] if architectures else config.get("model_type", "Unknown")
    experts = config.get("num_experts") or config.get("num_local_experts")
    is_moe = True if experts else (False if architecture != "Unknown" else None)
    context = config.get("max_position_embeddings") or config.get("model_max_length")
    params = card_data.get("parameters") or card_data.get("parameter_count") or "Unknown"
    active = card_data.get("active_parameters") or "Unknown"
    license_name = card_data.get("license") or next(
        (tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), "Unknown"
    )
    return ModelMetadata(
        repo_id=repo_id,
        name=repo_id.rsplit("/", 1)[-1],
        author=repo_id.split("/", 1)[0] if "/" in repo_id else "Unknown",
        downloads=getattr(info, "downloads", None),
        likes=getattr(info, "likes", None),
        license=str(license_name),
        architecture=str(architecture),
        parameter_count=str(params),
        active_parameters=str(active),
        context_length=int(context) if isinstance(context, int | float) else None,
        model_family=str(config.get("model_type", "Unknown")),
        is_moe=is_moe,
        description=_card_description(repo_id) if include_description else "Unknown",
        tags=tags,
    )


class HuggingFaceCatalog:
    def __init__(self, api: HfApi | None = None) -> None:
        self.api = api or HfApi()

    def search(self, query: str, limit: int = 12) -> list[ModelMetadata]:
        if not query.strip():
            return []
        # Prefer repositories already tagged as GGUF; conversion search is offered separately.
        results = self.api.list_models(search=query.strip(), filter="gguf", sort="downloads", limit=limit)
        return [parse_model_metadata(item) for item in results]

    def details(self, repo_id: str) -> ModelMetadata:
        info = self.api.model_info(repo_id, files_metadata=True)
        return parse_model_metadata(info, include_description=True)

    def discover_gguf(self, repo_id: str) -> list[GGUFFile]:
        info = self.api.model_info(repo_id, files_metadata=True)
        siblings = [s for s in (info.siblings or []) if s.rfilename.lower().endswith(".gguf")]
        grouped: dict[str, list[Any]] = defaultdict(list)
        singles: list[Any] = []
        for sibling in siblings:
            split = SPLIT_PATTERN.search(sibling.rfilename)
            if split:
                key = SPLIT_PATTERN.sub("", sibling.rfilename)
                grouped[key].append(sibling)
            else:
                singles.append(sibling)
        files: list[GGUFFile] = []
        for sibling in singles:
            size = getattr(sibling, "size", None) or 0
            files.append(GGUFFile(repo_id, sibling.rfilename, size, extract_quantization(sibling.rfilename)))
        for parts in grouped.values():
            ordered = sorted(parts, key=lambda item: item.rfilename)
            total_size = sum((getattr(item, "size", None) or 0) for item in ordered)
            files.append(
                GGUFFile(
                    repo_id,
                    ordered[0].rfilename,
                    total_size,
                    extract_quantization(ordered[0].rfilename),
                    tuple(item.rfilename for item in ordered),
                )
            )
        return sorted(files, key=lambda item: (item.size_bytes, item.filename))

    def find_conversions(self, repo_id: str, limit: int = 8) -> list[ModelMetadata]:
        model_name = repo_id.rsplit("/", 1)[-1]
        return self.search(f"{model_name} GGUF", limit=limit)


def metadata_dict(metadata: ModelMetadata) -> dict[str, Any]:
    data = asdict(metadata)
    data["architecture_type"] = "MoE" if metadata.is_moe else ("Dense" if metadata.is_moe is False else "Unknown")
    return data
