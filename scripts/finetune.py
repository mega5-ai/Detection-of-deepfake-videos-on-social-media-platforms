import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import librosa
import timm
from tqdm import tqdm  # live progress bar

# -----------------------------
# CONFIGURATION
# -----------------------------
FRAMES_ROOT = r"E:\PolyGlotFake\PolyGlotFake\frames"
AUDIO_ROOT = r"E:\PolyGlotFake\PolyGlotFake\audio"
BATCH_SIZE = 2
EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_PER_VIDEO = 8   # Increased from 4 to 8
MAX_AUDIO_LEN = 256    # fixed width for mel-spectrograms

# -----------------------------
# DATASET
# -----------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, frames_root, audio_root, transform=None, sr=16000, frames_per_video=8, max_audio_len=256):
        self.samples = []
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.sr = sr
        self.max_audio_len = max_audio_len

        # Loop through real and fake folders
        for label, cls in enumerate(["real", "fake"]):
            frame_cls_dir = os.path.join(frames_root, cls)
            audio_cls_dir = os.path.join(audio_root, cls)

            if not os.path.exists(frame_cls_dir) or not os.path.exists(audio_cls_dir):
                print(f"❌ Missing folder for class {cls}")
                continue

            # Go through subfolders like to_en, to_es, etc.
            for lang in os.listdir(frame_cls_dir):
                frame_lang_dir = os.path.join(frame_cls_dir, lang)
                audio_lang_dir = os.path.join(audio_cls_dir, lang)

                if not os.path.isdir(frame_lang_dir) or not os.path.exists(audio_lang_dir):
                    continue

                # Traverse video folders
                for video_name in os.listdir(frame_lang_dir):
                    video_path = os.path.join(frame_lang_dir, video_name)
                    if not os.path.isdir(video_path):
                        continue

                    # Find audio file that starts with video_name
                    audio_files = [f for f in os.listdir(audio_lang_dir) if f.startswith(video_name) and f.endswith(".wav")]
                    if not audio_files:
                        print(f"⚠️ No audio for video {video_name} in {lang} ({cls})")
                        continue

                    audio_path = os.path.join(audio_lang_dir, audio_files[0])
                    self.samples.append((video_path, audio_path, label))

        print(f"✅ Total samples found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_dir, audio_path, label = self.samples[idx]

        # --- Load Frames ---
        frames = sorted([
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])[:self.frames_per_video]

        imgs = []
        for f in frames:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        # If not enough frames, pad with last one
        while len(imgs) < self.frames_per_video:
            imgs.append(imgs[-1])
        imgs = torch.stack(imgs)

        # --- Load Audio ---
        y, _ = librosa.load(audio_path, sr=self.sr)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=64)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.tensor(mel).unsqueeze(0)  # [1, 64, T]

        # Pad/truncate
        if mel_tensor.shape[2] < self.max_audio_len:
            pad_len = self.max_audio_len - mel_tensor.shape[2]
            mel_tensor = nn.functional.pad(mel_tensor, (0, pad_len))
        else:
            mel_tensor = mel_tensor[:, :, :self.max_audio_len]

        return imgs, mel_tensor, torch.tensor(label)

# -----------------------------
# MODEL (Swin Transformer + Audio CNN)
# -----------------------------
class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
        self.cnn.head = nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.cnn.forward_features(dummy)
            self.visual_feat_dim = feat.mean([-2, -1]).shape[1]

        # Audio CNN
        self.audio_net = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 128),
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(self.visual_feat_dim + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, frames, audio):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)

        visual_feats = self.cnn.forward_features(frames)
        visual_feats = visual_feats.mean([-2, -1])
        visual_embeds = visual_feats.view(B, T, -1).mean(dim=1)

        audio_embeds = self.audio_net(audio)
        combined = torch.cat([visual_embeds, audio_embeds], dim=1)
        return self.fc(combined)

# -----------------------------
# TRAIN FUNCTION (Safe fine-tuning)
# -----------------------------
def train_finetune():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = DeepfakeDataset(
        FRAMES_ROOT,
        AUDIO_ROOT,
        transform=transform,
        frames_per_video=FRAMES_PER_VIDEO,
        max_audio_len=MAX_AUDIO_LEN
    )

    if len(dataset) == 0:
        raise ValueError("❌ No samples found. Check folders.")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DeepfakeModel().to(DEVICE)

    # Load pretrained weights
    state = torch.load("deepfake_model_swin_cnn_acc.pth", map_location=DEVICE)
    model.load_state_dict(state)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last swin transformer block, audio net, and classifier layers
    for param in model.cnn.layers[-1].parameters():
        param.requires_grad = True
    for param in model.audio_net.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # Class weights (increase weight of 'fake' class if needed)
    class_weights = torch.tensor([1.0, 1.3]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    print(f"🚀 Fine-tuning started on {DEVICE} with {len(dataset)} samples...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for frames, audio, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            frames, audio, labels = frames.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(frames, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total
        print(f"\n📘 Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "deepfake_model_finetuned.pth")
    print("✅ Fine-tuning complete. Model saved as deepfake_model_finetuned.pth")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    train_finetune()

