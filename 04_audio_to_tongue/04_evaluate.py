# -*- coding: utf-8 -*-
"""
04_audio_to_tongue / 04_evaluate.py  (3모델 비교 버전)

목적
----
02(Ridge, 오디오만) / 03(CNN, 오디오만) / 03b(CNN, 오디오+입모양)
세 모델을 정확히 같은 화자독립 테스트셋에서 비교 평가한다.

"입모양을 추가하면 혀 위치 예측이 더 좋아지는가?"라는 질문에
직접 답하는 단계다.
"""

import glob
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AudioToTongueCNN(nn.Module):
    """03_train_model.py와 동일 구조."""
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


def load_test_utterances():
    files = glob.glob(os.path.join(PROCESSED_DIR, "test", "*.npz"))
    utterances = []
    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        item = {
            "audio": data["audio"].astype(np.float32),
            "tongue": data["tongue"].astype(np.float32),
            "speaker": str(data["speaker"]),
        }
        if "video" in data:
            item["video"] = data["video"].astype(np.float32)
        utterances.append(item)
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
        preds.append(model.predict(x))
        targets.append(utt["tongue"][:n])
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_audio_cnn(utterances):
    ckpt_path = os.path.join(PROCESSED_DIR, "cnn_model_best.pt")
    if not os.path.exists(ckpt_path):
        print("[안내] 03_train_model.py를 먼저 실행하세요.")
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = AudioToTongueCNN(ckpt["n_mels"], ckpt["tongue_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for utt in utterances:
            n = min(len(utt["audio"]), len(utt["tongue"]))
            x = torch.from_numpy(utt["audio"][:n]).unsqueeze(0).to(DEVICE)
            pred = model(x).cpu().numpy()[0]
            preds.append(pred)
            targets.append(utt["tongue"][:n])
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_av_cnn(utterances):
    ckpt_path = os.path.join(PROCESSED_DIR, "cnn_model_best_av.pt")
    if not os.path.exists(ckpt_path):
        print("[안내] 03b_train_model_av.py를 먼저 실행하세요.")
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = AudioToTongueCNN(ckpt["in_channels"], ckpt["tongue_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for utt in utterances:
            if "video" not in utt:
                continue  # 입모양 특징 없는 (구버전) 데이터는 건너뜀
            n = min(len(utt["audio"]), len(utt["video"]), len(utt["tongue"]))
            av = np.concatenate([utt["audio"][:n], utt["video"][:n]], axis=1)
            x = torch.from_numpy(av).unsqueeze(0).to(DEVICE)
            pred = model(x).cpu().numpy()[0]
            preds.append(pred)
            targets.append(utt["tongue"][:n])
    if not preds:
        return None
    return np.concatenate(preds), np.concatenate(targets)


def report(name, result):
    if result is None:
        return None
    preds, targets = result
    print(f"\n=== {name} — 화자독립 테스트셋 최종 성능 ===")
    rs = []
    for d in range(targets.shape[1]):
        r, _ = pearsonr(targets[:, d], preds[:, d])
        rmse = float(np.sqrt(np.mean((targets[:, d] - preds[:, d]) ** 2)))
        print(f"  조음 파라미터 {d+1}번 축: r={r:.3f}, RMSE={rmse:.3f}")
        rs.append(r)
    return rs


def main():
    utterances = load_test_utterances()
    if not utterances:
        print("[안내] 테스트 데이터가 없습니다. 01_prepare_data.py를 먼저 실행하세요.")
        return
    print(f"테스트 발화 수: {len(utterances)} (화자: {sorted(set(u['speaker'] for u in utterances))})")

    r_ridge = report("02. Ridge 베이스라인 (오디오만, 문맥 없음)", evaluate_ridge(utterances))
    r_audio = report("03. CNN (오디오만, 문맥 포함)", evaluate_audio_cnn(utterances))
    r_av = report("03b. CNN (오디오+입모양, 문맥 포함)", evaluate_av_cnn(utterances))

    print("\n=== 종합 비교: 입모양 추가 효과 ===")
    if r_audio and r_av:
        for d, (ra, rav) in enumerate(zip(r_audio, r_av)):
            diff = rav - ra
            arrow = "▲ 개선" if diff > 0.02 else ("▼ 악화" if diff < -0.02 else "- 거의 동일")
            print(f"  조음 파라미터 {d+1}번 축: 오디오만 r={ra:.3f} → 오디오+입모양 r={rav:.3f}  ({arrow}, 차이 {diff:+.3f})")
        avg_diff = np.mean([rav - ra for ra, rav in zip(r_audio, r_av)])
        print(f"\n  평균 개선폭: {avg_diff:+.3f}")
        if avg_diff > 0.03:
            print("  -> 입모양 정보가 혀 위치 예측에 실질적으로 도움이 됩니다.")
            print("     실제 배포 시 웹캠(입모양)+마이크(음향)를 함께 04번 모델 입력으로 쓰는 것을 검토하세요.")
        elif avg_diff < -0.03:
            print("  -> 입모양을 추가했더니 오히려 나빠졌습니다. 데이터 부족으로 인한 과적합 가능성이 있습니다.")
            print("     데이터를 늘리거나, 입모양 특징 없이 오디오만 쓰는 현재 구조를 유지하는 것이 안전합니다.")
        else:
            print("  -> 입모양 추가 효과가 뚜렷하지 않습니다. 지금 데이터 규모에서는 오디오만으로도 충분하다는 뜻일 수 있습니다.")
            print("     TaL80 전체 등 더 큰 데이터로 재검증을 권장합니다.")
    else:
        print("  [안내] 두 CNN 결과가 모두 있어야 비교할 수 있습니다. 03/03b를 모두 실행했는지 확인하세요.")


if __name__ == "__main__":
    main()
