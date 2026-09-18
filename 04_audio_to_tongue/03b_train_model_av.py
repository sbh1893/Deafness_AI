# -*- coding: utf-8 -*-
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
CHUNK_LEN = 200
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AVTongueDataset(Dataset):
    """오디오+입모양을 이어붙인(concat) 입력을 반환한다."""

    def __init__(self, split_name, chunk_len=CHUNK_LEN):
        files = glob.glob(os.path.join(PROCESSED_DIR, split_name, "*.npz"))
        self.chunks = []
        for fpath in files:
            data = np.load(fpath, allow_pickle=True)
            if "video" not in data:
                continue  # 이전 버전(오디오만) 처리 결과는 건너뜀
            audio = data["audio"].astype(np.float32)
            video = data["video"].astype(np.float32)
            tongue = data["tongue"].astype(np.float32)
            n = min(len(audio), len(video), len(tongue))
            av = np.concatenate([audio[:n], video[:n]], axis=1)  # (T, n_mels+3)
            tongue = tongue[:n]
            for start in range(0, max(1, n - chunk_len + 1), chunk_len):
                a_chunk = av[start:start + chunk_len]
                t_chunk = tongue[start:start + chunk_len]
                if len(a_chunk) < chunk_len // 2:
                    continue
                self.chunks.append((a_chunk, t_chunk))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        av, tongue = self.chunks[idx]
        return torch.from_numpy(av), torch.from_numpy(tongue), len(av)


def collate_fn(batch):
    max_len = max(item[2] for item in batch)
    n_feat = batch[0][0].shape[1]
    n_tongue = batch[0][1].shape[1]
    av_batch = torch.zeros(len(batch), max_len, n_feat)
    tongue_batch = torch.zeros(len(batch), max_len, n_tongue)
    mask = torch.zeros(len(batch), max_len)
    for i, (av, tongue, length) in enumerate(batch):
        av_batch[i, :length] = av
        tongue_batch[i, :length] = tongue
        mask[i, :length] = 1.0
    return av_batch, tongue_batch, mask


class AudioVideoToTongueCNN(nn.Module):
    """03_train_model.py의 AudioToTongueCNN과 구조가 완전히 동일하고,
    입력 채널 수(in_channels)만 다르다."""

    def __init__(self, in_channels, tongue_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, tongue_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.net(x)
        h = h.transpose(1, 2)
        return self.head(h)


def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask.unsqueeze(-1)
    return diff2.sum() / (mask.sum() * pred.shape[-1] + 1e-8)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for av, tongue, mask in loader:
            av, tongue, mask = av.to(DEVICE), tongue.to(DEVICE), mask.to(DEVICE)
            pred = model(av)
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
        for av, tongue, mask in loader:
            av = av.to(DEVICE)
            pred = model(av).cpu().numpy()
            m = mask.numpy().astype(bool)
            for b in range(pred.shape[0]):
                preds.append(pred[b][m[b]])
                targets.append(tongue[b][m[b]].numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return [pearsonr(targets[:, d], preds[:, d])[0] for d in range(preds.shape[1])]


def main():
    print(f"device: {DEVICE}")
    train_ds = AVTongueDataset("train")
    val_ds = AVTongueDataset("val")
    if len(train_ds) == 0:
        print("[안내] video 특징이 포함된 학습 데이터가 없습니다. 01_prepare_data.py(입모양 버전)를 먼저 실행하세요.")
        return

    print(f"train 청크 수: {len(train_ds)}, val 청크 수: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    in_channels = train_ds.chunks[0][0].shape[1]
    tongue_dim = train_ds.chunks[0][1].shape[1]
    print(f"입력 채널 수: {in_channels} (오디오 멜채널 + 입모양 3)")

    model = AudioVideoToTongueCNN(in_channels=in_channels, tongue_dim=tongue_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_path = os.path.join(PROCESSED_DIR, "cnn_model_best_av.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader, optimizer=None)
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(),
                        "in_channels": in_channels, "tongue_dim": tongue_dim}, best_path)

    if len(val_ds) > 0:
        corrs = compute_correlation(model, val_loader)
        print("\n=== 검증셋 화자독립 상관계수 (오디오+입모양, 최종 epoch 기준) ===")
        for d, r in enumerate(corrs):
            print(f"  조음 파라미터 {d+1}번 축: r={r:.3f}")

    print(f"\n최적 모델(A+V) 저장 위치: {best_path}")


if __name__ == "__main__":
    main()
