"""Train the TODO.md MobileNetV3-Small + BiLSTM + CTC CAPTCHA model."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
CHAR_TO_INDEX = {character: index + 1 for index, character in enumerate(CHARSET)}
LABEL_RE = re.compile(r"^[a-z0-9]{1,32}$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def label_from_path(path: Path) -> str:
    label = path.stem.rsplit("_", 1)[-1]
    if not LABEL_RE.fullmatch(label):
        raise ValueError(f"cannot extract an alphanumeric label from {path.name!r}")
    return label


class CaptchaDataset(Dataset[tuple[Tensor, str]]):
    def __init__(self, directory: Path, augment: bool = False, limit: int | None = None):
        self.paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no captcha images found in {directory}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((52, 130)),
                transforms.RandomAffine(degrees=2, translate=(0.02, 0.02))
                if augment
                else transforms.Lambda(lambda image: image),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("L")
            return self.transform(image), label_from_path(path)


def collate_batch(batch: list[tuple[Tensor, str]]) -> tuple[Tensor, Tensor, Tensor]:
    images, labels = zip(*batch)
    encoded = [torch.tensor([CHAR_TO_INDEX[character] for character in label]) for label in labels]
    targets = torch.cat(encoded)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return torch.stack(images), targets, target_lengths


class CaptchaNet(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        first_conv = backbone.features[0][0]
        replacement = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                replacement.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                if first_conv.bias is not None:
                    replacement.bias.copy_(first_conv.bias)
        backbone.features[0][0] = replacement
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((2, 10))
        self.sequence = nn.LSTM(
            input_size=576 * 2,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Linear(256, len(CHARSET) + 1)

    def forward(self, images: Tensor) -> Tensor:
        features = self.pool(self.backbone(images))
        batch, channels, height, width = features.shape
        sequence = features.permute(0, 3, 1, 2).reshape(batch, width, channels * height)
        sequence, _ = self.sequence(sequence)
        return self.head(sequence).permute(1, 0, 2)


def set_backbone_frozen(model: CaptchaNet, frozen: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = not frozen


def decode_greedy(logits: Tensor) -> list[str]:
    predictions = logits.argmax(dim=2).transpose(0, 1).tolist()
    decoded: list[str] = []
    for sequence in predictions:
        characters: list[str] = []
        previous = 0
        for index in sequence:
            if index != 0 and index != previous:
                characters.append(CHARSET[index - 1])
            previous = index
        decoded.append("".join(characters))
    return decoded


def evaluate(model: CaptchaNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    exact = 0
    character_total = 0
    character_correct = 0
    sample_total = 0
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            logits = model(images.to(device))
            predictions = decode_greedy(logits)
            offset = 0
            for prediction, length in zip(predictions, target_lengths.tolist()):
                target = "".join(CHARSET[index - 1] for index in targets[offset : offset + length].tolist())
                offset += length
                exact += prediction == target
                character_total += len(target)
                character_correct += sum(
                    predicted == expected
                    for predicted, expected in zip(prediction, target)
                )
                sample_total += 1
    return {
        "exact_match": exact / sample_total if sample_total else 0.0,
        "character_accuracy": character_correct / character_total if character_total else 0.0,
        "samples": float(sample_total),
    }


def train_epoch(
    model: CaptchaNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, targets, target_lengths in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        input_lengths = torch.full(
            (images.shape[0],), logits.shape[0], dtype=torch.long, device=device
        )
        loss = criterion(logits.log_softmax(2), targets, input_lengths, target_lengths.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.shape[0]
    return total_loss / len(loader.dataset)


def export_onnx(model: CaptchaNet, path: Path, device: torch.device) -> None:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 1, 52, 130), device=device)
    torch.onnx.export(
        model,
        sample,
        path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {1: "batch"}},
        opset_version=18,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/collected"))
    parser.add_argument("--output", type=Path, default=Path("outputs/captcha_mobilenet_ctc.pt"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cuda" if device_name == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    data_dir = args.data_dir.resolve()
    train_set = CaptchaDataset(data_dir / "auto_labeled", augment=True, limit=args.max_train_samples)
    train_eval_set = CaptchaDataset(
        data_dir / "auto_labeled", augment=False, limit=args.max_train_samples
    )
    test_set = CaptchaDataset(
        data_dir / "reviewed", augment=False, limit=args.max_test_samples
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    train_eval_loader = DataLoader(
        train_eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = CaptchaNet(pretrained=args.pretrained).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, float]] = []
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint.get("epoch", len(history)))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"resumed checkpoint={args.resume} start_epoch={start_epoch}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        set_backbone_frozen(model, epoch <= args.freeze_epochs)
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = evaluate(model, train_eval_loader, device)
        metrics = evaluate(model, test_loader, device)
        scheduler.step()
        record = {
            "epoch": float(epoch),
            "loss": loss,
            "train_exact_match": train_metrics["exact_match"],
            "train_character_accuracy": train_metrics["character_accuracy"],
            "test_exact_match": metrics["exact_match"],
            "test_character_accuracy": metrics["character_accuracy"],
            "test_samples": metrics["samples"],
        }
        history.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "charset": CHARSET,
                "input_size": [1, 52, 130],
                "history": history,
            },
            args.output,
        )
        print(
            f"epoch={epoch:03d} loss={loss:.4f} "
            f"train_exact={train_metrics['exact_match']:.2%} "
            f"train_char={train_metrics['character_accuracy']:.2%} "
            f"test_exact={metrics['exact_match']:.2%} "
            f"test_char={metrics['character_accuracy']:.2%}"
        )

    if args.onnx:
        export_onnx(model, args.onnx, device)
    print(json.dumps({"checkpoint": str(args.output), "test": history[-1]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
