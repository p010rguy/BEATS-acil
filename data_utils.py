from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class ESC50SplitDataset(Dataset):
    def __init__(
        self,
        splits: dict,
        audio_dir: Path,
        use_split: str = "train",
        sample_rate: int = 16000,
        class_ids: list[int] | None = None,
        class_to_idx: dict[int, int] | None = None,
    ) -> None:
        if use_split not in {"train", "test"}:
            raise ValueError("use_split must be 'train' or 'test'")

        self.sample_rate = sample_rate
        self.audio_dir = audio_dir

        if class_ids is None:
            class_ids = list(splits["base_classes"])
        class_ids = list(class_ids)
        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(class_ids)}
        else:
            self.class_to_idx = dict(class_to_idx)
        class_ids_set = set(class_ids)

        if use_split == "train":
            files_by_target = splits["train_files_by_target"]
        else:
            files_by_target = splits["test_files_by_target"]

        items: list[tuple[Path, int]] = []
        for target_str, files in files_by_target.items():
            target = int(target_str)
            if target not in class_ids_set:
                continue
            for fname in files:
                items.append((audio_dir / fname, self.class_to_idx[target]))

        if not items:
            raise RuntimeError("No audio files found for base classes")

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, target = self.items[idx]
        if not path.exists():
            raise FileNotFoundError(f"Missing audio file: {path}")
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        except Exception:
            data, sr = sf.read(path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data).unsqueeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.sample_rate,
            )
        waveform = waveform.squeeze(0)
        return waveform, target


def pad_collate(batch: list[tuple[torch.Tensor, int]]):
    waveforms, targets = zip(*batch)
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)
    batch_size = len(waveforms)

    audio = torch.zeros(batch_size, max_len, dtype=waveforms[0].dtype)#创建max_len全0张量
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)#创建max_len布尔True张量

    for i, w in enumerate(waveforms):
        audio[i, : w.shape[0]] = w #将第i条语音塞入全0张量
        padding_mask[i, : w.shape[0]] = False #将第i条语音塞入，对应位置改为false，为不是padding——mask

    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return audio, padding_mask, targets_tensor
