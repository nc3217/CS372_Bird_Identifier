# Computing inference timing, metric, and error analysis parameters

import json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(OUTPUT_DIR)
OUT.mkdir(parents=True, exist_ok=True)

WARMUP_ITERS = 5
MISCLASS_PER_CLASS = 5
MIN_SUPPORT = 10

print("Using device:", DEVICE)
print("Saving outputs to:", OUT)


if "label_idx" not in test_df.columns:
    test_df = test_df.copy()
    test_df["label_idx"] = test_df["label"].map(label2idx)


test_tf = get_transforms(img_size=IMG_SIZE, train=False)

class CSVImageDataset(Dataset):
    def __init__(self, df, img_col="image_path", label_col="label_idx", transform=None):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.img_col]).convert("RGB")
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)["image"]
        label = int(row[self.label_col])
        return img, label, row[self.img_col]

test_ds = CSVImageDataset(test_df, transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)

print("Test samples:", len(test_ds))

# ensure model is loaded
if "model" not in globals():
    print("No model in memory: loading best.pth...")
    best_path = OUT / "best.pth"
    ckpt = torch.load(best_path, map_location=DEVICE)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(idx2label))
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.to(DEVICE)

model.eval()

# run inference
print("\nRunning inference over test set...")
ys, preds, probs, paths = [], [], [], []

with torch.no_grad():
    for imgs, labels, img_paths in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        pr = logits.argmax(dim=1).cpu().numpy()

        probs.append(p)
        preds.append(pr)
        ys.append(labels.numpy())
        paths.extend(img_paths)

probs = np.concatenate(probs)
y_pred = np.concatenate(preds)
y_true = np.concatenate(ys)


print("\nMeasuring inference timing...")
if DEVICE.type == "cuda":
    with torch.no_grad():
        it = 0
        for imgs, _, _ in test_loader:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            it += 1
            if it >= WARMUP_ITERS:
                break

batch_times = []
total_images = 0
start_all = time.perf_counter()

with torch.no_grad():
    for imgs, _, _ in test_loader:
        imgs = imgs.to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        _ = model(imgs)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        batch_times.append((t_end - t_start, imgs.shape[0]))
        total_images += imgs.shape[0]

end_all = time.perf_counter()
total_time = end_all - start_all
throughput = total_images / total_time

latencies = np.concatenate([np.repeat(bt / bs, bs) for bt, bs in batch_times])
lat_mean = latencies.mean()
lat_median = np.median(latencies)
lat_p95 = np.percentile(latencies, 95)

print("\nINFERENCE TIMING SUMMARY:")
print(f"  Total images: {total_images}, Time: {total_time:.3f}s")
print(f"  Throughput: {throughput:.2f} img/s")
print(f"  Mean latency: {lat_mean*1000:.2f} ms")
print(f"  Median latency: {lat_median*1000:.2f} ms")
print(f"  p95 latency: {lat_p95*1000:.2f} ms")

# compute and output metrics
acc_top1 = accuracy_score(y_true, y_pred)
prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

top5_acc = None
if probs.shape[1] >= 5:
    top5_preds = np.argsort(probs, axis=1)[:, -5:]
    top5_acc = np.mean([1 if y_true[i] in top5_preds[i] else 0 for i in range(len(y_true))])

print("\nMETRICS (TEST SET):")
print(f"  Top-1 Accuracy : {acc_top1:.4f}")
print(f"  Macro F1       : {f1_macro:.4f}")
print(f"  Precision      : {prec_macro:.4f}")
print(f"  Recall         : {rec_macro:.4f}")
print(f"  Top-5 Accuracy : {top5_acc:.4f}")


summary = {
    "device": str(DEVICE),
    "total_images": int(total_images),
    "total_time_s": float(total_time),
    "throughput_img_per_s": float(throughput),
    "lat_mean_s": float(lat_mean),
    "lat_median_s": float(lat_median),
    "lat_p95_s": float(lat_p95),
    "top1_acc": float(acc_top1),
    "macro_f1": float(f1_macro),
    "precision_macro": float(prec_macro),
    "recall_macro": float(rec_macro),
    "top5_acc": float(top5_acc),
}
Path(OUT/"inference_summary.json").write_text(json.dumps(summary, indent=2))

print("Saved inference_summary.json")

# per-class f1
print("\nComputing per-class F1...")
classes = sorted(idx2label.keys())
per_class = []

for c in classes:
    support = int((y_true == c).sum())
    if support < MIN_SUPPORT:
        continue
    f1c = f1_score((y_true == c).astype(int),
                   (y_pred == c).astype(int),
                   average="binary",
                   zero_division=0)
    per_class.append((c, idx2label[c], f1c, support))

pc_df = pd.DataFrame(per_class, columns=["idx", "label", "f1", "support"])
pc_df = pc_df.sort_values("f1").reset_index(drop=True)
pc_df.to_csv(OUT/"per_class_f1.csv", index=False)
display(pc_df.head(30))

# worst-class misclassifications
print("\nVisualizing misclassified examples for worst classes...")
worst = pc_df.head(5)

plt.figure(figsize=(18, 12))
plot_idx = 1
for _, row in worst.iterrows():
    cls_idx = int(row["idx"])
    mis_idx = np.where((y_true == cls_idx) & (y_pred != cls_idx))[0]
    sel = mis_idx[:MISCLASS_PER_CLASS]

    for mid in sel:
        img_path = test_df.iloc[mid]["image_path"]
        true_lbl = idx2label[int(y_true[mid])]
        pred_lbl = idx2label[int(y_pred[mid])]
        prob_pred = float(probs[mid, int(y_pred[mid])])
        prob_true = float(probs[mid, int(y_true[mid])])

        try:
            im = Image.open(img_path).convert("RGB").resize((256, 256))
        except:
            continue

        ax = plt.subplot(len(worst), MISCLASS_PER_CLASS, plot_idx)
        ax.imshow(im); ax.axis("off")
        ax.set_title(f"T:{true_lbl[:30]}\nP:{pred_lbl[:30]}\nP:{prob_pred:.2f} T:{prob_true:.2f}", fontsize=8)
        plot_idx += 1

plt.tight_layout()
plt.savefig(OUT/"misclassified_examples.png", dpi=150)
plt.show()

# three examples of misclassifications in more detail
print("\nGenerating 3 detailed misclassifications...\n")

mis_idx_all = np.where(y_true != y_pred)[0]
if len(mis_idx_all) == 0:
    print("No misclassifications found.")
else:
    chosen = np.random.choice(mis_idx_all, size=min(3, len(mis_idx_all)), replace=False)
    fig = plt.figure(figsize=(14, 6))

    for i, mid in enumerate(chosen, start=1):
        img_path = test_df.iloc[mid]["image_path"]
        im = Image.open(img_path).convert("RGB").resize((300, 300))

        true_lbl = idx2label[int(y_true[mid])]
        pred_lbl = idx2label[int(y_pred[mid])]

        top5_idx = probs[mid].argsort()[-5:][::-1]
        top5_labels = [idx2label[k] for k in top5_idx]
        top5_scores = [float(probs[mid][k]) for k in top5_idx]

        ax = fig.add_subplot(1, 3, i)
        ax.imshow(im); ax.axis("off")
        ax.set_title(f"Example {i}", fontsize=12)

        print(f"EXAMPLE {i}")
        print(f"Image: {img_path}")
        print(f"TRUE LABEL:      {true_lbl}")
        print(f"PREDICTED LABEL: {pred_lbl}")
        print("Top-5 Predictions:")
        for lbl, score in zip(top5_labels, top5_scores):
            print(f"   {lbl:45s} {score:.4f}")
        print("\n")

    plt.tight_layout()
    plt.savefig(OUT/"example_misclassifications.png", dpi=150)
    plt.show()

print("\nDone! Analysis saved to:", OUT)

