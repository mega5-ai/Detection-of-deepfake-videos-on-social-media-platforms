import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from train import DeepfakeDataset, DeepfakeModel, FRAMES_ROOT, AUDIO_ROOT, DEVICE, FRAMES_PER_VIDEO, MAX_AUDIO_LEN, BATCH_SIZE

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_model(model_path="mode1.pth"):
    print("🔍 Loading model and dataset...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = DeepfakeDataset(
        FRAMES_ROOT,
        AUDIO_ROOT,
        transform=transform,
        frames_per_video=FRAMES_PER_VIDEO,
        max_audio_len=MAX_AUDIO_LEN
    )

    # Use small subset for faster evaluation (optional)
    if len(dataset) > 200:
        dataset = Subset(dataset, range(200))

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = DeepfakeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print("🧪 Evaluating model...")
    with torch.no_grad():
        for frames, audio, labels in tqdm(loader, desc="Evaluating", unit="batch"):
            frames, audio = frames.to(DEVICE), audio.to(DEVICE)
            outputs = model(frames, audio)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary')
    rec = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print("\n✅ Evaluation Results:")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

# -----------------------------
# RUN EVALUATION
# -----------------------------
if __name__ == "__main__":
    evaluate_model()
