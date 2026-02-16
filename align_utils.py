from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


from model_utils import BEATsWithHead, extract_pooled_features


def select_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if preference == "mps-cuda":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")


def select_acc_dtype(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float32
    return torch.float64

def init_w_fe(feat_dim: int, fe_dim: int, device: torch.device, dtype: torch.dtype, seed: int = 2026):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W_fe = torch.randn(feat_dim, fe_dim, generator=g, dtype=dtype).to(device)
    # 可选：做个缩放，避免尺度爆炸（很重要）
    W_fe = W_fe / (feat_dim ** 0.5)
    return W_fe

def features_expansion(model: BEATsWithHead, audio, padding_mask, W_fe=None, use_relu=True, normalize=True, acc_dtype=torch.float32):
    feats = extract_pooled_features(model.beats, audio, padding_mask).to(acc_dtype)  # [B, 768]
    if W_fe is not None:
        feats = feats @ W_fe  # [B, fe_dim]
        if use_relu:
            feats = F.relu(feats)
    if normalize:
        feats = F.normalize(feats, dim=1)
    return feats

def cls_align_beats(
    train_loader: DataLoader,
    model: BEATsWithHead,
    device: torch.device,
    num_classes: int,
    rg: float = 1e-3,
    W_fe = None
) -> torch.Tensor:
    model.eval()
    feat_dim = (W_fe.size(1) if W_fe is not None else model.classifier.weight.size(1))
    acc_dtype = select_acc_dtype(device)
    auto_cor = torch.zeros(feat_dim, feat_dim, device=device, dtype=acc_dtype) #D,D
    crs_cor = torch.zeros(feat_dim, num_classes, device=device, dtype=acc_dtype)#D,C
    with torch.no_grad():
        for audio, padding_mask, targets in train_loader:
            audio = audio.to(device)
            padding_mask = padding_mask.to(device)
            targets = targets.to(device)
            feats = features_expansion(model, audio, padding_mask, W_fe).to(acc_dtype)
            label_onehot = torch.nn.functional.one_hot(targets, num_classes).to(acc_dtype)
            auto_cor += feats.t() @ feats #Xfe_T * Xfe
            crs_cor += feats.t() @ label_onehot #Xfe_t * Y = Q

    eye = torch.eye(feat_dim, device=device, dtype=acc_dtype) # I
    reg = rg * eye #gamma * I
    R = torch.linalg.inv((auto_cor + reg).cpu()).to(device) #(Xfe_T * Xfe + gamma*I)^-1
    #formula 4 初始化
    W = R @ crs_cor #W_FCN
    model.classifier.weight = torch.nn.parameter.Parameter((0.9 * W.t()).float())
    return R, W


def il_align_beats(
    train_loader: DataLoader,
    model: BEATsWithHead,
    device: torch.device,
    num_classes: int,
    R: torch.Tensor,
    W,
    repeat: int = 1,
    W_fe = None,
) -> torch.Tensor:
    model.eval()
    acc_dtype = select_acc_dtype(device)
    W = W.to(acc_dtype)
    print("W shape: ", W.shape)
    R = R.to(acc_dtype)
    print("R shape: ", R.shape)
    with torch.no_grad():
        for _ in range(repeat):
            for audio, padding_mask, targets in train_loader:
                audio = audio.to(device)
                padding_mask = padding_mask.to(device)
                targets = targets.to(device)
                feats = features_expansion(model, audio, padding_mask, W_fe=W_fe, acc_dtype=acc_dtype)  # [B, feat_dim]
                label_onehot = torch.nn.functional.one_hot(targets, num_classes).to(acc_dtype)

                eye = torch.eye(feats.size(0), device=device, dtype=acc_dtype) #I
                #I+Xfe_k*R*Xfe_kT
                mid = torch.linalg.pinv(eye + feats @ R @ feats.t())
                #formula 10
                R = R - R @ feats.t() @ mid @ feats @ R
                #formula 9
                W = W + R @ feats.t() @ (label_onehot - feats @ W)

    model.classifier.weight = torch.nn.parameter.Parameter(W.t().float())
    return R, W


def evaluate_accuracy(
    loader: DataLoader,
    model: BEATsWithHead,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for audio, padding_mask, targets in loader:
            audio = audio.to(device)
            padding_mask = padding_mask.to(device)
            targets = targets.to(device)
            logits = model(audio, padding_mask)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return correct / max(total, 1)

def evaluate_accuracy_acil(
    loader: DataLoader,
    model: BEATsWithHead,
    device: torch.device,
    W: torch.Tensor,
    W_fe: torch.Tensor | None = None,
    use_relu: bool = True,
    normalize: bool = True,
) -> float:
    model.eval()
    acc_dtype = select_acc_dtype(device)

    W = W.to(device=device, dtype=acc_dtype)

    correct = 0
    total = 0
    with torch.no_grad():
        for audio, padding_mask, targets in loader:
            audio = audio.to(device)
            padding_mask = padding_mask.to(device)
            targets = targets.to(device)

            feats = features_expansion(
                model,
                audio,
                padding_mask,
                W_fe=W_fe,
                use_relu=use_relu,
                normalize=normalize,
                acc_dtype=acc_dtype,
            )
            logits = feats @ W  # [B, C]
            preds = logits.argmax(dim=1)

            correct += (preds == targets).sum().item()
            total += targets.numel()

    return correct / max(total, 1)
