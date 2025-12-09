# Sample image classification - can be used to test model with sample_bird.jpg

import torch
import timm
import numpy as np
from PIL import Image
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "tf_efficientnet_b3"
IMG_SIZE = 300
OUTPUT_DIR = "/content/drive/MyDrive/bird_models"


import json
label_map_path = Path(OUTPUT_DIR) / "label_map.json"
idx2label = {int(k): v for k, v in json.loads(label_map_path.read_text()).items()}
label2idx = {v: k for k, v in idx2label.items()}

n_classes = len(idx2label)

print("Loaded label maps:", n_classes, "classes")



# transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

infer_tf = get_transforms()


# load model
best_path = Path(OUTPUT_DIR) / "best.pth"

print("Loading model from:", best_path)

ckpt = torch.load(best_path, map_location=DEVICE)
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=n_classes)

state = ckpt.get("model_state", ckpt)
model.load_state_dict(state, strict=False)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# prediction function
def predict_image(img_path, topk=5):
    """
    Runs inference on a single image path and returns top-k predictions.
    """

    # Load image
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)

    # Apply transforms
    tensor = infer_tf(image=arr)["image"].unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Top-k
    idxs = probs.argsort()[::-1][:topk]
    results = [(idx2label[int(i)], float(probs[i])) for i in idxs]

    return results

# show the prediction
def show_prediction(img_path, topk=5):
    """
    Prints results cleanly and displays the image.
    """
    from IPython.display import display

    img = Image.open(img_path)
    display(img)

    preds = predict_image(img_path, topk=topk)

    print("\n🔎 Top Predictions:")
    for rank, (label, score) in enumerate(preds, start=1):
        print(f"{rank}. {label} — {score:.4f}")


# Put your own filename here:
test_img = "/content/sample_bird.jpg"

print("\nRunning model on:", test_img)
show_prediction(test_img, topk=5)

