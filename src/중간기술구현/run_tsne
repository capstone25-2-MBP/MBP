# 0. import
# 1. transform 정의
# 2. 이미지 로딩
# 3. ResNet 로드
# 4. latent vector 추출
#========================
# 5. domain-wise mixup
# 6. t-SNE
# 7. visualization

import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# 1. 설정
# =========================
DATA_ROOT = "."
BATCH_SIZE = 16
DEVICE = "cpu"   # GPU 없어도 OK

# =========================
# 2. 이미지 변환 정의
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# 3. 이미지 로딩
# =========================
images = []
labels = []

print("[INFO] Loading images...")

for domain in os.listdir(DATA_ROOT):
    domain_path = os.path.join(DATA_ROOT, domain)
    if not os.path.isdir(domain_path):
        continue

    for fname in os.listdir(domain_path):
        if fname.lower().endswith(".jpg"):
            img_path = os.path.join(domain_path, fname)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)

            images.append(img)
            labels.append(domain)

images = torch.stack(images).to(DEVICE)

print(f"[INFO] Total images: {len(images)}")
print(f"[INFO] Domains: {set(labels)}")

# =========================
# 4. ResNet 로드
# =========================
model = models.resnet18(pretrained=True)
model.eval()
model.to(DEVICE)

# =========================
# 5. Latent vector 추출
# =========================
features = []

def hook_fn(module, input, output):
    features.append(output.detach().cpu())

handle = model.layer4.register_forward_hook(hook_fn)

print("[INFO] Extracting latent vectors...")

with torch.no_grad():
    for i in tqdm(range(0, len(images), BATCH_SIZE)):
        batch = images[i:i + BATCH_SIZE]
        _ = model(batch)

handle.remove()

# =========================
# 6. Feature 정리
# =========================
latent = torch.cat(features, dim=0)
latent = latent.mean(dim=[2, 3])   # Global Average Pooling
latent = latent.numpy()

print(f"[INFO] Latent shape: {latent.shape}")  # (N, 512)

# =========================
# 7. t-SNE
# =========================
print("[INFO] Running t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    random_state=42
)

embeds_2d = tsne.fit_transform(latent)

# =========================
# 8. 시각화
# =========================
print("[INFO] Visualizing...")

domains = sorted(list(set(labels)))
color_map = plt.cm.tab10(np.linspace(0, 1, len(domains)))

plt.figure(figsize=(8, 8))

for domain, color in zip(domains, color_map):
    idx = [i for i, l in enumerate(labels) if l == domain]
    plt.scatter(
        embeds_2d[idx, 0],
        embeds_2d[idx, 1],
        label=domain,
        alpha=0.7,
        color=color
    )

plt.legend()
plt.title("t-SNE of Medical Image Latent Vectors")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.tight_layout()
plt.show()
