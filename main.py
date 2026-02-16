from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from align_utils import (
    cls_align_beats,
    il_align_beats,
    select_acc_dtype,
    select_device,
    evaluate_accuracy_acil,
    init_w_fe
)
from cli_args import build_parser
from data_utils import ESC50SplitDataset, pad_collate
from downloader import download_esc50
from model_utils import (
    BEATsWithHead,
    expand_classifier,
    load_beats_backbone,
    load_beats_model,
    maybe_resume_checkpoint,
)
from split_esc50 import ESC_ROOT, make_splits


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        download_esc50(Path(args.dest), force=args.force)
        esc_root = ESC_ROOT
        if not esc_root.exists():
            raise FileNotFoundError(f"ESC-50 root not found: {esc_root}")
        splits = make_splits(
            esc_root=esc_root,
            test_fold=1,
            seed=2026,
        )
        out_path = Path("esc50_25_5x5_splits.json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        print(f"Saved splits to {out_path.resolve()}")

        beats_checkpoint_path = Path("checkpoints/BEATs_iter3_plus_AS2M.pt")
        if args.basetraining and beats_checkpoint_path.exists():
            device = select_device(args.device)
            beats = load_beats_model(beats_checkpoint_path, device).to(device)
            print(f"BEATs model loaded from: {beats_checkpoint_path}")
            train_dataset = ESC50SplitDataset(
                splits=splits,
                audio_dir=esc_root / "audio",
                use_split="train",
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0,
                collate_fn=pad_collate,
            )
            val_dataset = ESC50SplitDataset(
                splits=splits,
                audio_dir=esc_root / "audio",
                use_split="test",
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0,
                collate_fn=pad_collate,
            )
            model = BEATsWithHead(beats, num_classes=25).to(device)
            model.train()
            print(f"Training mode: {model.training}, device: {device}")

            optimizer = torch.optim.Adam(
                [
                    {"params": model.beats.parameters(), "lr": 1e-5},
                    {"params": model.classifier.parameters(), "lr": 1e-3},
                ]
            )
            loss_fn = torch.nn.CrossEntropyLoss()

            for epoch in range(1, 5 + 1):
                epoch_start = time.perf_counter()
                total_loss = 0.0
                num_batches = 0
                for audio, padding_mask, targets in train_loader:
                    audio = audio.to(device)
                    padding_mask = padding_mask.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    logits = model(audio, padding_mask)
                    loss = loss_fn(logits, targets)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / max(num_batches, 1)
                model.eval()
                print(f"Eval mode: {model.training} (should be False)")
                correct = 0
                total = 0
                with torch.no_grad():
                    for audio, padding_mask, targets in val_loader:
                        audio = audio.to(device)
                        padding_mask = padding_mask.to(device)
                        targets = targets.to(device)
                        logits = model(audio, padding_mask)
                        preds = logits.argmax(dim=1)
                        correct += (preds == targets).sum().item()
                        total += targets.numel()
                acc = correct / max(total, 1)
                epoch_time = time.perf_counter() - epoch_start
                print(
                    f"Epoch {epoch}/5 - loss: {avg_loss:.4f} - val_acc: {acc:.4f} "
                    f"- time: {epoch_time:.1f}s"
                )
                model.train()
                print(f"Training mode: {model.training} (should be True)")

            ckpt_out = Path("checkpoints/beats_finetuned_base25.pt")
            ckpt_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "cfg": model.beats.cfg.__dict__,
                    "model": model.state_dict(),
                    "num_classes": 25,
                },
                ckpt_out,
            )
            print(f"Saved fine-tuned checkpoint to: {ckpt_out}")

            backbone_out = Path("checkpoints/beats_base25_backbone.pt")
            torch.save(
                {
                    "cfg": model.beats.cfg.__dict__,
                    "model": model.beats.state_dict(),
                },
                backbone_out,
            )
            print(f"Saved BEATs backbone checkpoint to: {backbone_out}")
        elif args.basetraining:
            print(
                f"BEATs checkpoint not found: {beats_checkpoint_path}",
                file=sys.stderr,
            )

        if not args.incremental:
            return 0
        
        if args.incremental:

            backbone_ckpt_path = Path("checkpoints/beats_base25_backbone.pt")
            if backbone_ckpt_path.exists():
                device = select_device(args.device)
                beats = load_beats_backbone(backbone_ckpt_path, device)
                beats.to(device)
                beats.eval()
                for p in beats.parameters():
                    p.requires_grad = False
                print(f"BEATs backbone loaded from: {backbone_ckpt_path}")
                base_classes = list(splits["base_classes"])[: args.baseclass]
                incremental_groups = splits["incremental_classes"][: args.phase]
                if incremental_groups:
                    for idx, group in enumerate(incremental_groups, start=1):
                        count = len(base_classes) + sum(
                            len(g) for g in incremental_groups[:idx]
                        )
                        print(f"Seen classes after phase {idx}: {count}")
                model = BEATsWithHead(beats, num_classes=len(base_classes)).to(device)
                model.eval()
                print(f"Alignment mode: {model.training}, device: {device}")
                maybe_resume_checkpoint(model, args.resume, device)

                base_class_to_idx = {cid: idx for idx, cid in enumerate(base_classes)}
                base_train_dataset = ESC50SplitDataset(
                    splits=splits,
                    audio_dir=esc_root / "audio",
                    use_split="train",
                    class_ids=base_classes,
                    class_to_idx=base_class_to_idx,
                )
                base_train_loader = DataLoader(
                    base_train_dataset,
                    batch_size=4,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=pad_collate,
                )
                base_val_dataset = ESC50SplitDataset(
                    splits=splits,
                    audio_dir=esc_root / "audio",
                    use_split="test",
                    class_ids=base_classes,
                    class_to_idx=base_class_to_idx,
                )
                base_val_loader = DataLoader(
                    base_val_dataset,
                    batch_size=4,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=pad_collate,
                )

                acc_cil: list[float] = []
                forget_rate: list[float] = []
                if args.fe_dim is None:
                    W_fe = None
                else:
                    W_fe = init_w_fe(
                        feat_dim=model.classifier.weight.size(1),
                        fe_dim=args.fe_dim,
                        device=device,
                        dtype=select_acc_dtype(device),
                        seed=2026
                    )          
                R, W = cls_align_beats(
                    train_loader=base_train_loader,
                    model=model,
                    device=device,
                    num_classes=len(base_classes),
                    rg=args.rg,
                    W_fe=W_fe
                )
                base_acc = evaluate_accuracy_acil(base_val_loader, model, device, W, W_fe)
                acc_cil.append(base_acc)
                print(f"Base phase acc: {base_acc:.4f}")
                
                for phase_idx, inc_classes in enumerate(incremental_groups, start=1):
                    print(f"Phase {phase_idx} classes: {inc_classes}")
                    new_num_classes = len(base_classes) + sum(
                        len(g) for g in incremental_groups[:phase_idx]
                    )
                    expand_classifier(model, new_num_classes)
                
                    # 同步扩展 W: [fe_dim, old_C] -> [fe_dim, new_C]
                    if W.size(1) < new_num_classes:
                        W_new = torch.zeros(
                            W.size(0), new_num_classes,
                            device=W.device, dtype=W.dtype
                        )
                        W_new[:, :W.size(1)] = W
                        W = W_new
                    
                    
                    seen_classes = list(base_classes)
                    for g in incremental_groups[:phase_idx]:
                        seen_classes.extend(g)
                    phase_class_to_idx = {
                        cid: idx for idx, cid in enumerate(seen_classes)
                    }

                    inc_train_dataset = ESC50SplitDataset(
                        splits=splits,
                        audio_dir=esc_root / "audio",
                        use_split="train",
                        class_ids=inc_classes,
                        class_to_idx=phase_class_to_idx,
                    )
                    inc_train_loader = DataLoader(
                        inc_train_dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=pad_collate,
                    )

                    if isinstance(R, tuple):
                        if len(R) == 2:
                            R, W = R
                        else:
                            raise ValueError(f"Unexpected R tuple length: {len(R)}")
                        
                    R, W= il_align_beats(
                        train_loader=inc_train_loader,
                        model=model,
                        device=device,
                        num_classes=new_num_classes,
                        R=R,
                        W=W,
                        repeat=1,
                        W_fe=W_fe
                    )

                    val_dataset = ESC50SplitDataset(
                        splits=splits,
                        audio_dir=esc_root / "audio",
                        use_split="test",
                        class_ids=seen_classes,
                        class_to_idx=phase_class_to_idx,
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=4,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=pad_collate,
                    )

                    acc = evaluate_accuracy_acil(val_loader, model, device, W, W_fe)
                    acc_cil.append(acc)
                    if acc_cil:
                        base_acc_now = evaluate_accuracy_acil(base_val_loader, model, device, W, W_fe)
                        forget_rate.append(acc_cil[0] - base_acc_now)

                    print(
                        f"Phase {phase_idx}/{len(incremental_groups)} "
                        f"- acc: {acc:.4f} "
                        f"- base_now: {base_acc_now:.4f} "
                        f"- forget: {forget_rate[-1]:.4f}"
                    )

            if acc_cil:
                avg = sum(acc_cil) / len(acc_cil)
                print(f"Average accuracy: {avg:.4f}")
            else:
                print("Average accuracy: n/a (no phases evaluated)")

            ckpt_out = Path("checkpoints/beats_base25_incremental.pt")
            ckpt_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "cfg": model.beats.cfg.__dict__,
                    "model": model.state_dict(),
                    "num_classes": model.classifier.out_features,
                },
                ckpt_out,
            )
            print(f"Saved incremental checkpoint to: {ckpt_out}")
        else:
            print(
                f"BEATs backbone checkpoint not found: {backbone_ckpt_path}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
