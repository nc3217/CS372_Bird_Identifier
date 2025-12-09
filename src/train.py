#Import data

import kagglehub
path = kagglehub.dataset_download("sharansmenon/inatbirds100k")
print("Dataset path:", path)

import os
import pandas as pd

data = []

root_path = os.path.join(path, "birds_train_small")



# Length of Dataset

for species in os.listdir(root_path):
    species_path = os.path.join(root_path, species)

    if not os.path.isdir(species_path):
        continue

    for img in os.listdir(species_path):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(species_path, img)
            data.append([img_path, species])

df = pd.DataFrame(data, columns=["image_path", "label"])

len(df)



# Splitting the Data

from sklearn.model_selection import train_test_split


train_df, test_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=42)

train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)

print(len(train_df), len(val_df), len(test_df))



# Random Baseline Model

import numpy as np
import pandas as pd

def class_prior_random_baseline(train_df, test_df, label_col="label", seed=None):

    if seed is not None:
        np.random.seed(seed)

    # compute class frequencies in the training data
    class_counts = train_df[label_col].value_counts().sort_index()
    class_probs = class_counts / class_counts.sum()


    y_true = test_df[label_col].values

    # predictions
    y_pred = np.random.choice(
        a=class_counts.index.values,   # class IDs
        size=len(test_df),
        p=class_probs.values
    )

    # accuracy
    accuracy = (y_pred == y_true).mean()

    return accuracy

baseline_acc = class_prior_random_baseline(train_df, test_df, seed=42)
print("Class-prior random baseline accuracy:", baseline_acc)



# Generating Embeddings for Pre-trained CLIP Model

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

SAVE_PATH = "/content/drive/MyDrive/bird_clip_embeddings.npy"
SAVE_LABELS = "/content/drive/MyDrive/bird_clip_labels.npy"

# load CLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", weights_only=False).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


image_paths = train_df['image_path'].tolist()
labels = train_df['label'].values

BATCH_SIZE = 64
total_embeddings = []

# computing and saving embeddings
for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    batch_images = [Image.open(Path(p)) for p in batch_paths]

    inputs = clip_processor(images=batch_images, return_tensors="pt").to(device)

    with torch.no_grad():
        batch_emb = clip_model.get_image_features(**inputs)

    batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
    total_embeddings.append(batch_emb.cpu().numpy())

image_embeddings = np.concatenate(total_embeddings, axis=0)

np.save(SAVE_PATH, image_embeddings)
np.save(SAVE_LABELS, labels)

print("Saved embeddings to:", SAVE_PATH)
print("Saved labels to:", SAVE_LABELS)
print("Embedding shape:", image_embeddings.shape)



# Creating and Evaluating Pre-Trained CLIP Logistic Regression Model

from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib, json

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# paths for embeddings and labels
EMB_PATH = "/content/drive/MyDrive/bird_clip_embeddings.npy"
LABEL_PATH = "/content/drive/MyDrive/bird_clip_labels.npy"

X = np.load(EMB_PATH)
y_raw = np.load(LABEL_PATH, allow_pickle=True)

print("Loaded embeddings:", X.shape)
print("Loaded labels:", y_raw.shape)

# convert labels to integers
if y_raw.dtype.kind in {'U', 'S', 'O'}:
    unique = sorted(list(set(y_raw.tolist())))
    label2idx = {lab: i for i, lab in enumerate(unique)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    y = np.array([label2idx[s] for s in y_raw], dtype=np.int32)
else:
    y = y_raw.astype(np.int32)
    idx2label = {i: str(i) for i in range(int(y.max()) + 1)}

print("Num classes:", len(idx2label))

# normalize embeddings
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# split training and validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

print("Training logistic regression on", X_tr.shape[0], "samples")


clf = LogisticRegression(
    max_iter=300,
    solver='lbfgs',
    C=1.0,
    n_jobs=-1,
    multi_class='auto'
)

clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, y_pred))

print(classification_report(
    y_val, y_pred,
    target_names=[idx2label[i] for i in sorted(idx2label)]
))

# save model and label map
OUTDIR = Path("/content/drive/MyDrive/bird_models")
OUTDIR.mkdir(exist_ok=True)

joblib.dump(clf, OUTDIR / "logistic_baseline.joblib")
json.dump(idx2label, open(OUTDIR / "label_map.json", "w"))

print("Saved model to:", OUTDIR / "logistic_baseline.joblib")
print("Saved label map to:", OUTDIR / "label_map.json")



# EfficientNet Model Training

!pip install -q timm albumentations==1.3.0

import os
import math
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# hyperparameters
MODEL_NAME = "tf_efficientnet_b3"
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-4
WEIGHT_DECAY = 1e-2
OUTPUT_DIR = "/content/drive/MyDrive/bird_models"
USE_WEIGHTED_SAMPLER = True
PRETRAINED = True
NUM_WORKERS = 4
PIN_MEMORY = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Convert labels to a contiguous integer range
all_labels = pd.concat([train_df['label'], test_df['label'], val_df['label']]).unique()
all_labels = sorted(list(all_labels))
label2idx = {lab: i for i, lab in enumerate(all_labels)}
idx2label = {i:lab for lab,i in label2idx.items()}
n_classes = len(label2idx)
print("Num classes:", n_classes)

train_df = train_df.copy()
test_df  = test_df.copy()
val_df = val_df.copy()
train_df['label_idx'] = train_df['label'].map(label2idx)
test_df['label_idx']  = test_df['label'].map(label2idx)
val_df['label_idx'] = val_df['label'].map(label2idx)

# transform images in dataset
def get_transforms(img_size=IMG_SIZE, train=True):
    if train:
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.6,1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])

class CSVImageDataset(Dataset):
    def __init__(self, df, img_col='image_path', label_col='label_idx', transform=None):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.img_col]
        # PIL open and convert
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        label = int(row[self.label_col])
        return img, label

train_ds = CSVImageDataset(train_df, transform=get_transforms(IMG_SIZE, train=True))
val_ds   = CSVImageDataset(val_df,  transform=get_transforms(IMG_SIZE, train=False))

# sampler to help mitigate any class imbalance
train_sampler = None
if USE_WEIGHTED_SAMPLER:
    counts = train_df['label_idx'].value_counts().sort_index().values
    class_weights = 1.0 / (counts + 1e-12)
    sample_weights = train_df['label_idx'].map(lambda x: class_weights[x]).values
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    shuffle = False
else:
    shuffle = True

# data loaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=shuffle if train_sampler is None else False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print("Train samples:", len(train_ds), "Val samples:", len(val_ds))

# create model
model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=n_classes)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler()

# training function
def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.detach().cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return avg_loss, acc, f1

# validation function
@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    pbar = tqdm(loader, desc="val", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.detach().cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return avg_loss, acc, f1, y_true, y_pred

# training loop
best_val_f1 = -1.0
for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
    val_loss, val_acc, val_f1, y_true, y_pred = validate(model, val_loader, device, criterion)
    scheduler.step()
    print(f"Train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f}")
    print(f"Val   loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")
    # Save checkpoint
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_f1": val_f1
    }
    torch.save(ckpt, os.path.join(OUTPUT_DIR, f"ckpt_epoch{epoch}.pth"))
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(ckpt, os.path.join(OUTPUT_DIR, "best.pth"))
        print("Saved best model.")
    if epoch % 3 == 0:
        print("Sample classification report (first 30 classes):")
        try:
            print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(min(n_classes,30))], zero_division=0))
        except Exception:
            pass
    if train_acc >= 0.61:
      break
print("Training finished. Best val F1:", best_val_f1)
