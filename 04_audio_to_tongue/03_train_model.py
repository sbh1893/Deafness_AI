# -*- coding: utf-8 -*-
"""
04_audio_to_tongue / 03_train_model.py

목적
----
02번 베이스라인(Ridge 회귀, 프레임 단위·문맥 없음)보다 나은 성능을 위해,
오디오 특징의 시간적 문맥(앞뒤 프레임 흐름)을 보는 경량 1D CNN 모델을
학습한다.

설계 근거
--------
- 발음은 순간적인 스냅샷이 아니라 연속적인 움직임이므로, 앞뒤 프레임을
  같이 보는 것이 이론적으로 더 유리하다. (02번 베이스라인은 프레임
  하나만 보고 예측하므로 이 정보를 활용하지 못한다.)
- 다만 처음부터 무거운 모델(Transformer 등)을 쓰지 않고, 작은 1D CNN
  스택으로 시작한다 — 데이터 규모가 크지 않을 수 있어 과적합 위험이
  있기 때문에, 03_semg에서도 강조했던 '작은 데이터로 버티는' 원칙을
  동일하게 적용한다.

실행 전 준비
-----------
pip install torch --index-url https://download.pytorch.org/whl/cpu
(GPU가 있다면 공식 홈페이지에서 CUDA 버전에 맞는 명령어로 설치)
"""

import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
CHUNK_LEN = 200          # 한 번에 학습에 넣을 프레임 길이 (약 2초, hop=10ms 기준)
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# 데이터셋
# ----------------------------------------------------------------------
class TongueDataset(Dataset):
    """발화를 CHUNK_LEN 길이로 잘라 (오디오, 조음파라미터) 쌍을 반환한다."""

    def __init__(self, split_name, chunk_len=CHUNK_LEN):
        files = glob.glob(os.path.join(PROCESSED_DIR, split_name, "*.npz"))
        self.chunks = []
        for fpath in files:
            data = np.load(fpath, allow_pickle=True)
            audio = data["audio"].astype(np.float32)
            tongue = data["tongue"].astype(np.float32)
            n = min(len(audio), len(tongue))
            audio, tongue = audio[:n], tongue[:n]
            for start in range(0, max(1, n - chunk_len + 1), chunk_len):
                a_chunk = audio[start:start + chunk_len]
                t_chunk = tongue[start:start + chunk_len]
                if len(a_chunk) < chunk_len // 2:
                    continue  # 너무 짧은 조각은 버림
                self.chunks.append((a_chunk, t_chunk))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio, tongue = self.chunks[idx]
        return torch.from_numpy(audio), torch.from_numpy(tongue), len(audio)


def collate_fn(batch):
    """가변 길이 청크를 배치 내 최대 길이에 맞춰 0으로 패딩한다."""
    max_len = max(item[2] for item in batch)
    n_mels = batch[0][0].shape[1]
    n_tongue = batch[0][1].shape[1]
    audio_batch = torch.zeros(len(batch), max_len, n_mels)
    tongue_batch = torch.zeros(len(batch), max_len, n_tongue)
    mask = torch.zeros(len(batch), max_len)
    for i, (audio, tongue, length) in enumerate(batch):
        audio_batch[i, :length] = audio
        tongue_batch[i, :length] = tongue
        mask[i, :length] = 1.0
    return audio_batch, tongue_batch, mask


# ----------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------
class AudioToTongueCNN(nn.Module):
    """경량 1D CNN: (B, T, n_mels) -> (B, T, tongue_dim)"""

    def __init__(self, n_mels=40, tongue_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, tongue_dim)

    def forward(self, x):
        # x: (B, T, n_mels) -> Conv1d는 (B, C, T)를 기대하므로 transpose
        x = x.transpose(1, 2)
        h = self.net(x)
        h = h.transpose(1, 2)  # (B, T, hidden)
        return self.head(h)


# ----------------------------------------------------------------------
# 학습 / 평가 루프
# ----------------------------------------------------------------------
def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask.unsqueeze(-1)
    return diff2.sum() / (mask.sum() * pred.shape[-1] + 1e-8)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for audio, tongue, mask in loader:
            audio, tongue, mask = audio.to(DEVICE), tongue.to(DEVICE), mask.to(DEVICE)
            pred = model(audio)
            loss = masked_mse(pred, tongue, mask)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def compute_correlation(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for audio, tongue, mask in loader:
            audio = audio.to(DEVICE)
            pred = model(audio).cpu().numpy()
            m = mask.numpy().astype(bool)
            for b in range(pred.shape[0]):
                preds.append(pred[b][m[b]])
                targets.append(tongue[b][m[b]].numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    correlations = []
    for d in range(preds.shape[1]):
        r, _ = pearsonr(targets[:, d], preds[:, d])
        correlations.append(r)
    return correlations


def main():
    print(f"device: {DEVICE}")
    train_ds = TongueDataset("train")
    val_ds = TongueDataset("val")
    if len(train_ds) == 0:
        print("[안내] 학습 데이터가 없습니다. 먼저 01_prepare_data.py를 실행하세요.")
        return

    print(f"train 청크 수: {len(train_ds)}, val 청크 수: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    n_mels = train_ds.chunks[0][0].shape[1]
    tongue_dim = train_ds.chunks[0][1].shape[1]
    model = AudioToTongueCNN(n_mels=n_mels, tongue_dim=tongue_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_path = os.path.join(PROCESSED_DIR, "cnn_model_best.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader, optimizer=None)
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(),
                        "n_mels": n_mels, "tongue_dim": tongue_dim}, best_path)

    if len(val_ds) > 0:
        corrs = compute_correlation(model, val_loader)
        print("\n=== 검증셋 화자독립 상관계수 (최종 epoch 기준) ===")
        for d, r in enumerate(corrs):
            print(f"  조음 파라미터 {d+1}번 축: r={r:.3f}")

    print(f"\n최적 모델 저장 위치: {best_path}")


if __name__ == "__main__":
    main()
