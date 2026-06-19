"""4개 이미지 분류 모델에서 feature를 추출해 proxy 데이터를 생성한다.

각 모델의 분류기 직전 embedding과 예측값을 NPZ/CSV로 저장한다.
모든 모델을 실행한 경우에는 embedding을 이어 붙인 combined proxy도 저장한다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent))
from train_backbones import LABELS, create_model  # noqa: E402


BACKBONES = ["densenet121", "resnet50", "efficientnet_b0", "vit_b_16"]


class ProxyImageDataset(Dataset):
    def __init__(self, df, image_root, transform):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.label_columns = [label for label in LABELS if label in df.columns]

    def __len__(self):
        return len(self.df)

    def resolve_image(self, image_path):
        raw = Path(str(image_path))
        candidates = [
            raw,
            self.image_root / raw,
            self.image_root / raw.stem / raw.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"이미지를 찾을 수 없습니다: {image_path}\n"
            f"확인한 경로: {[str(path) for path in candidates]}"
        )

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(self.resolve_image(row["image_path"])).convert("RGB")
        if self.label_columns:
            labels = row[self.label_columns].to_numpy(dtype=np.float32)
        else:
            labels = np.empty(0, dtype=np.float32)
        return self.transform(image), torch.from_numpy(labels), str(row["image_path"])


def build_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def classifier_layer(model, backbone):
    """분류기 입력을 hook할 마지막 linear layer를 반환한다."""
    if backbone == "densenet121":
        return model.classifier
    if backbone == "resnet50":
        return model.fc
    if backbone == "efficientnet_b0":
        return model.classifier[1]
    if backbone == "vit_b_16":
        return model.heads.head
    raise ValueError(f"지원하지 않는 backbone: {backbone}")


def find_checkpoint(model_dir, backbone, tier):
    candidates = sorted(Path(model_dir).glob(f"{backbone}_{tier}_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"{model_dir}에서 {backbone}의 {tier} checkpoint를 찾지 못했습니다."
        )
    if len(candidates) > 1:
        print(
            f"[경고] {backbone}/{tier} checkpoint가 여러 개라 "
            f"{candidates[-1].name}을 사용합니다.",
            flush=True,
        )
    return candidates[-1]


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone = checkpoint["backbone"]
    image_size = int(checkpoint.get("args", {}).get("image_size", 224))
    model = create_model(backbone, pretrained=False, image_size=image_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, image_size


def extract_proxy(model, backbone, loader, device):
    feature_buffer = []

    def save_classifier_input(_module, inputs, _output):
        feature_buffer.append(inputs[0].detach().cpu())

    hook = classifier_layer(model, backbone).register_forward_hook(save_classifier_input)
    all_features = []
    all_logits = []
    all_labels = []
    all_paths = []

    try:
        with torch.inference_mode():
            for images, labels, image_paths in loader:
                feature_buffer.clear()
                logits = model(images.to(device, non_blocking=True))
                if len(feature_buffer) != 1:
                    raise RuntimeError(
                        f"{backbone} feature hook 호출 횟수가 예상과 다릅니다: "
                        f"{len(feature_buffer)}"
                    )
                all_features.append(feature_buffer[0].numpy())
                all_logits.append(logits.detach().cpu().numpy())
                if labels.shape[1] > 0:
                    all_labels.append(labels.numpy())
                all_paths.extend(image_paths)
    finally:
        hook.remove()

    result = {
        "features": np.concatenate(all_features).astype(np.float32),
        "logits": np.concatenate(all_logits).astype(np.float32),
        "image_paths": np.asarray(all_paths),
    }
    result["probabilities"] = (
        1.0 / (1.0 + np.exp(-result["logits"]))
    ).astype(np.float32)
    if all_labels:
        result["labels"] = np.concatenate(all_labels).astype(np.float32)
    return result


def save_model_proxy(output_dir, backbone, checkpoint_path, result, label_columns):
    model_dir = output_dir / backbone
    model_dir.mkdir(parents=True, exist_ok=True)

    npz_path = model_dir / "proxy_data.npz"
    np.savez_compressed(
        npz_path,
        backbone=np.asarray(backbone),
        checkpoint=np.asarray(str(checkpoint_path)),
        **result,
    )

    metadata = pd.DataFrame(
        {
            "proxy_index": np.arange(len(result["image_paths"])),
            "image_path": result["image_paths"],
        }
    )
    for index, label in enumerate(LABELS):
        metadata[f"prob_{label}"] = result["probabilities"][:, index]
    if "labels" in result:
        for index, label in enumerate(label_columns):
            metadata[label] = result["labels"][:, index]

    csv_path = model_dir / "proxy_metadata.csv"
    metadata.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(
        f"{backbone}: {result['features'].shape} -> {npz_path}",
        flush=True,
    )
    return npz_path


def save_combined_proxy(output_dir, results, label_columns):
    reference = results[BACKBONES[0]]
    for backbone in BACKBONES[1:]:
        if not np.array_equal(reference["image_paths"], results[backbone]["image_paths"]):
            raise RuntimeError(f"{backbone}의 이미지 순서가 다른 모델과 일치하지 않습니다.")

    features = np.concatenate(
        [results[backbone]["features"] for backbone in BACKBONES],
        axis=1,
    ).astype(np.float32)
    probabilities = np.stack(
        [results[backbone]["probabilities"] for backbone in BACKBONES],
        axis=1,
    ).astype(np.float32)

    combined = {
        "features": features,
        "probabilities": probabilities,
        "image_paths": reference["image_paths"],
        "backbones": np.asarray(BACKBONES),
    }
    if "labels" in reference:
        combined["labels"] = reference["labels"]

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    npz_path = combined_dir / "proxy_data.npz"
    np.savez_compressed(npz_path, **combined)

    metadata = pd.DataFrame(
        {
            "proxy_index": np.arange(len(reference["image_paths"])),
            "image_path": reference["image_paths"],
        }
    )
    if "labels" in reference:
        for index, label in enumerate(label_columns):
            metadata[label] = reference["labels"][:, index]
    metadata.to_csv(
        combined_dir / "proxy_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"combined: {features.shape} -> {npz_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="4개 backbone에서 이미지 feature 기반 proxy 데이터를 생성합니다."
    )
    parser.add_argument("--csv", default="test.csv")
    parser.add_argument("--image-root", default="(1000)verified_data")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--output-dir", default="outputs/proxy_data")
    parser.add_argument("--tier", choices=["low", "middle", "high"], default="high")
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=BACKBONES)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    if "image_path" not in df.columns:
        raise ValueError(f"{args.csv}에 image_path 열이 없습니다.")
    label_columns = [label for label in LABELS if label in df.columns]

    results = {}
    for backbone in args.backbones:
        checkpoint_path = find_checkpoint(args.model_dir, backbone, args.tier)
        model, _checkpoint, image_size = load_model(checkpoint_path, device)
        dataset = ProxyImageDataset(
            df=df,
            image_root=args.image_root,
            transform=build_transform(image_size),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        print(f"{backbone}: {checkpoint_path.name} feature 추출 중...", flush=True)
        result = extract_proxy(model, backbone, loader, device)
        results[backbone] = result
        save_model_proxy(
            output_dir,
            backbone,
            checkpoint_path,
            result,
            label_columns,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(args.backbones) == len(BACKBONES) and set(args.backbones) == set(BACKBONES):
        save_combined_proxy(output_dir, results, label_columns)


if __name__ == "__main__":
    main()
