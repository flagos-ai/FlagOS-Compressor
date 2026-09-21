from __future__ import annotations

import json
import shutil
import struct
from collections.abc import Iterator
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


SAFETENSORS_INDEX = "model.safetensors.index.json"


class HfSafetensorsCheckpoint:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.index_path = self.model_path / SAFETENSORS_INDEX
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as f:
                self.index = json.load(f)
        else:
            single_file = self.model_path / "model.safetensors"
            if not single_file.exists():
                raise FileNotFoundError(
                    f"Missing {self.index_path} and {single_file}"
                )
            with safe_open(single_file, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
            self.index = {
                "metadata": {},
                "weight_map": {name: single_file.name for name in keys},
            }
        self.weight_map: dict[str, str] = dict(self.index["weight_map"])

    def shard_files(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    def iter_shards(self) -> Iterator[tuple[str, Path]]:
        for shard in self.shard_files():
            yield shard, self.model_path / shard

    def load_shard(self, shard: str):
        return load_file(self.model_path / shard, device="cpu")

    def load_tensor(self, tensor_name: str):
        shard = self.weight_map[tensor_name]
        with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(tensor_name)

    def iter_tensor_metadata(self):
        """Read safetensors headers without mapping or allocating weight data."""
        for shard, path in self.iter_shards():
            with path.open("rb") as handle:
                prefix = handle.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"Truncated safetensors header: {path}")
                length = struct.unpack("<Q", prefix)[0]
                if length > 100_000_000 or length > path.stat().st_size - 8:
                    raise ValueError(f"Invalid safetensors header length: {path}")
                header = json.loads(handle.read(length))
            for name, metadata in header.items():
                if name != "__metadata__":
                    yield name, shard, metadata

    def save_shard(self, output_path: str | Path, shard: str, state_dict: dict) -> None:
        save_file(state_dict, str(Path(output_path) / shard))

    def copy_auxiliary_files(self, output_path: str | Path) -> None:
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        for item in self.model_path.iterdir():
            if item.name.endswith(".safetensors") or item.name == SAFETENSORS_INDEX:
                continue
            if item.name in {".download_complete.json", ".download-partial"}:
                continue
            target = output / item.name
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)

    def write_index(
        self,
        output_path: str | Path,
        weight_map: dict[str, str],
        *,
        total_size: int | None = None,
    ) -> None:
        metadata = dict(self.index.get("metadata", {}))
        if total_size is not None:
            metadata["total_size"] = total_size
        new_index = {
            "metadata": metadata,
            "weight_map": weight_map,
        }
        with (Path(output_path) / SAFETENSORS_INDEX).open("w", encoding="utf-8") as f:
            json.dump(new_index, f, indent=2)
