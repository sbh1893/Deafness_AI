# -*- coding: utf-8 -*-
"""
04_audio_to_tongue / 04_evaluate.py

목적
----
02번(Ridge 베이스라인)과 03번(CNN 모델)을 같은 테스트셋(학습에 전혀
없던 화자)에서 비교 평가한다. "복잡한 모델을 만드는 게 실제로 의미가
있었는가"를 확인하는 단계다.
"""

import glob
import os

import joblib
import numpy as np
import torch
from scipy.stats import pearsonr

from importlib import import_module

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 03_train_model.py의 모델 클래스를 재사용하기 위해 동적 임포트
train_module = import_module("03_train_model".replace("-", "_")) if False else None


def load_cnn_model():
    ckpt_path = os.path.join(PROCESSED_DIR, "cnn_model_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    # 03_train_model.py와 동일한 구조를 그대로 정의 (파일명이 숫자로 시작해 import가 번거로우므로 재정의)
    import torch.nn as nn

    class AudioToTongueCNN(nn.Module):
        def __init__(self, n_mels, tongue_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_mels, hidden, kernel_size=5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.BatchNorm1d(hidden), nn.ReLU(),
            )
            self.head = nn.Linear(hidden, tongue_dim)

        def forward(self, x):
            x = x.transpose(1, 2)
            h = self.net(x)
            h = h.transpose(1, 2)
            return self.head(h)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = AudioToTongueCNN(ckpt["n_mels"], ckpt["tongue_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_test_utterances():
    files = glob.glob(os.path.join(PROCESSED_DIR, "test", "*.npz"))
    utterances = []
    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        utterances.append({
            "audio": data["audio"].astype(np.float32),
            "tongue": data["tongue"].astype(np.float32),
            "speaker": str(data["speaker"]),
            "name": os.path.basename(fpath),
        })
    return utterances


def evaluate_ridge(utterances):
    baseline_path = os.path.join(PROCESSED_DIR, "baseline_ridge.joblib")
    if not os.path.exists(baseline_path):
        print("[안내] 02_baseline_eigentongue.py를 먼저 실행하세요.")
        return None
    ckpt = joblib.load(baseline_path)
    model, mean, std = ckpt["model"], ckpt["mean"], ckpt["std"]

    preds, targets = [], []
    for utt in utterances:
        n = min(len(utt["audio"]), len(utt["tongue"]))
        x = (utt["audio"][:n] - mean) / std
        pred = model.predict(x)
        preds.append(pred)
        targets.append(utt["tongue"][:n])
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_cnn(model, utterances):
    if model is None:
        print("[안내] 03_train_model.py를 먼저 실행하세요.")
        return None
    preds, targets = [], []
    with torch.no_grad():
        for utt in utterances:
            n = min(len(utt["audio"]), len(utt["tongue"]))
            x = torch.from_numpy(utt["audio"][:n]).unsqueeze(0).to(DEVICE)
            pred = model(x).cpu().numpy()[0]
            preds.append(pred)
            targets.append(utt["tongue"][:n])
    return np.concatenate(preds), np.concatenate(targets)


def report(name, preds, targets):
    if preds is None:
        return
    print(f"\n=== {name} — 화자독립 테스트셋 최종 성능 ===")
    for d in range(targets.shape[1]):
        r, _ = pearsonr(targets[:, d], preds[:, d])
        rmse = float(np.sqrt(np.mean((targets[:, d] - preds[:, d]) ** 2)))
        print(f"  조음 파라미터 {d+1}번 축: r={r:.3f}, RMSE={rmse:.3f}")


def main():
    utterances = load_test_utterances()
    if not utterances:
        print("[안내] 테스트 데이터가 없습니다. 01_prepare_data.py를 먼저 실행하세요.")
        return
    print(f"테스트 발화 수: {len(utterances)} (화자: {sorted(set(u['speaker'] for u in utterances))})")

    ridge_result = evaluate_ridge(utterances)
    report("02. Ridge 베이스라인 (문맥 없음)", *ridge_result if ridge_result else (None, None))

    cnn_model = load_cnn_model()
    cnn_result = evaluate_cnn(cnn_model, utterances)
    report("03. CNN 모델 (문맥 포함)", *cnn_result if cnn_result else (None, None))

    print("\n=== 해석 가이드 ===")
    print("- CNN 모델의 r이 Ridge보다 뚜렷이 높다면: 시간적 문맥이 실제로 도움이 된다는 뜻")
    print("- 두 모델 다 r이 0에 가깝다면: 오디오만으로 혀 위치 추정 자체가 이 데이터에서 어렵다는 뜻")
    print("  → 이 경우 화면 피드백의 신뢰도 표시(점선/흐림)를 더 강하게 적용해야 함")


if __name__ == "__main__":
    main()
