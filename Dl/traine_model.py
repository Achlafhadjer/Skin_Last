import glob
import json
import os
import random
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

# ====================== 1. Configuration ======================
SEED = 42
num_epochs = 30
batch_size = 16
patience_early_stopping = 6
val_size = 0.2

model_save_path = "checkpoints"
checkpoint_file = "last_training_state.pth"
class_names_file = "class_names.json"
best_model_file = "best_model.pth"

HAM10000_ROOT = os.getenv("HAM10000_ROOT", "/kaggle/input")
METADATA_FILENAME = "HAM10000_metadata.csv"

os.makedirs(model_save_path, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# HAM10000 official class mapping
DX_TO_NAME = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = [DX_TO_NAME[x] for x in CLASS_ORDER]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


def find_ham10000_metadata(root_dir):
    candidates = glob.glob(os.path.join(root_dir, "**", METADATA_FILENAME), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"Impossible de trouver {METADATA_FILENAME} sous {root_dir}. "
            "Définis HAM10000_ROOT vers le dossier du dataset."
        )
    return candidates[0]


def resolve_image_path(image_id, search_root):
    pattern = os.path.join(search_root, "**", f"{image_id}.jpg")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image introuvable pour image_id={image_id}")


def basic_cleaning(pil_img):
    # Correction orientation EXIF + léger auto-contraste
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return ImageOps.autocontrast(img, cutoff=1)


class HAM10000Dataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        img = Image.open(item["image_path"])
        img = basic_cleaning(img)
        if self.transform:
            img = self.transform(img)
        return img, item["label"]


print("--- Chargement HAM10000 ---")
metadata_path = find_ham10000_metadata(HAM10000_ROOT)
dataset_root = os.path.dirname(metadata_path)
print(f"Metadata: {metadata_path}")

df = pd.read_csv(metadata_path)
df = df[df["dx"].isin(CLASS_ORDER)].copy()
df["label"] = df["dx"].map(CLASS_TO_IDX)
df["image_path"] = df["image_id"].apply(lambda x: resolve_image_path(x, dataset_root))

train_df, val_df = train_test_split(
    df,
    test_size=val_size,
    random_state=SEED,
    stratify=df["label"],
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

with open(class_names_file, "w", encoding="utf-8") as f:
    json.dump({"classes": CLASS_NAMES}, f, ensure_ascii=False, indent=2)

print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")
print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

# ====================== 2. Transformations ======================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.5, 2.0)),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_records = train_df[["image_path", "label"]].to_dict("records")
val_records = val_df[["image_path", "label"]].to_dict("records")

train_dataset = HAM10000Dataset(train_records, transform=train_transform)
val_dataset = HAM10000Dataset(val_records, transform=val_transform)

# ====================== 3. Sampler + loaders ======================
train_labels = train_df["label"].tolist()
label_counts = Counter(train_labels)
class_weights = torch.tensor(
    [1.0 / label_counts[i] for i in range(len(CLASS_NAMES))], dtype=torch.float
)
sample_weights = class_weights[torch.tensor(train_labels)]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

_workers = int(os.getenv("NUM_WORKERS", "2"))
_pin = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=_workers,
    pin_memory=_pin,
    persistent_workers=_workers > 0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_workers,
    pin_memory=_pin,
)

# ====================== 4. Modèle ======================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(
    [
        {"params": model.features.parameters(), "lr": 7e-5},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ],
    weight_decay=1e-4,
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# ====================== 5. Resume ======================
start_epoch = 0
best_val_acc = 0.0
epochs_without_improvement = 0

if os.path.exists(checkpoint_file):
    print(f"--- Checkpoint trouvé ({checkpoint_file}), chargement... ---")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
    print(f"--- Reprise à l'epoch {start_epoch + 1} | best_val={best_val_acc:.2f}% ---")
else:
    print("--- Aucun checkpoint, nouvel entraînement ---")


def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            total += lbls.size(0)
            correct += (preds == lbls).sum().item()
    model.train()
    return 100.0 * correct / total if total else 0.0


# ====================== 6. Entraînement ======================
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_correct, running_total = 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for imgs, lbls in loop:
        imgs, lbls = imgs.to(device), lbls.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_total += lbls.size(0)
        running_correct += (preds == lbls).sum().item()
        loop.set_postfix(loss=f"{loss.item():.4f}", train_acc=f"{100.0 * running_correct / running_total:.2f}%")

    train_acc = 100.0 * running_correct / running_total
    val_acc = evaluate(val_loader)
    scheduler.step(val_acc)
    print(f"Epoch {epoch + 1}: train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}%")

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "epochs_without_improvement": epochs_without_improvement,
            "class_names": CLASS_NAMES,
        },
        checkpoint_file,
    )

    epoch_path = f"{model_save_path}/model_epoch_{epoch + 1}_val_{val_acc:.2f}.pth"
    torch.save(model.state_dict(), epoch_path)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_file)
        print(f"--- Nouveau meilleur modèle: {best_val_acc:.2f}% ---")
    else:
        epochs_without_improvement += 1
        print(f"--- Pas d'amélioration ({epochs_without_improvement}/{patience_early_stopping}) ---")
        if epochs_without_improvement >= patience_early_stopping:
            print("--- Early stopping activé ---")
            break

print("Entraînement terminé. Meilleure accuracy validation:", f"{best_val_acc:.2f}%")
