# -*- coding: utf-8 -*-
"""
02_mic_acoustic / audio_feature_capture.py

목적
----
마이크만으로 음향 신호 기반 조음 판별이 얼마나 가능한지 사전 검증하기 위한
최소 프로토타입입니다. (하드웨어 추가 구매 없이 지금 바로 실행 가능)

동작 요약
--------
1. 사용자가 콘솔에 목표 음소 라벨을 입력하면, 짧은 시간(RECORD_SECONDS) 동안
   마이크로 녹음합니다.
2. librosa로 피치(F0), 스펙트럴 특징(centroid/bandwidth), MFCC를 추출해
   CSV에 저장합니다. (6장에서 언급한 '비음은 음향학적으로 뚜렷한 특징을
   남긴다'는 근거를 실제로 확인해보기 위함)
3. 'q'를 입력해 종료하면, 지금까지 모은 데이터로 간단한 분류기를 학습해
   baseline 정확도를 보여줍니다.

자세한 설명은 상위 폴더의 WIKI.md 를 참고하세요.
"""

import csv
import os
import warnings
from datetime import datetime

import numpy as np
import sounddevice as sd
import librosa

warnings.filterwarnings("ignore")  # librosa의 자잘한 경고 숨김

# ----------------------------------------------------------------------
# 설정값
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000
RECORD_SECONDS = 1.5
CSV_PATH = os.path.join(os.path.dirname(__file__), "audio_features.csv")
N_MFCC = 13

LABEL_GUIDE = {
    "m": "ㅁ (비음, 1순위)",
    "n": "ㄴ (비음, 1순위)",
    "ng": "ㅇ받침 (비음, 1순위)",
    "g": "ㄱ (연구개음, 2순위)",
    "k": "ㅋ (연구개음, 2순위)",
    "kk": "ㄲ (연구개음, 2순위)",
}


def extract_audio_features(y, sr):
    """librosa로 피치/스펙트럴/MFCC 특징을 뽑는다.

    주의: spectral_centroid / spectral_bandwidth는 정식 포먼트(F1/F2/F3)
    추출이 아니라 '대략적인 음색 변화'를 보는 간이 지표입니다. 정밀한
    포먼트 분석이 필요하면 2단계에서 parselmouth(Praat) 등을 검토하세요.
    """
    if np.max(np.abs(y)) < 1e-4:
        return None  # 무음(마이크 문제 등)

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced_f0 = f0[~np.isnan(f0)]
    mean_pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)

    feat = {
        "mean_pitch": mean_pitch,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
    }
    for i, v in enumerate(mfcc_mean):
        feat[f"mfcc{i+1}"] = float(v)
    return feat


def ensure_csv_header(feature_keys):
    is_new = not os.path.exists(CSV_PATH)
    if is_new:
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label"] + feature_keys)


def record_one_label(label):
    print(f"  -> '{LABEL_GUIDE.get(label, label)}' 녹음 시작! {RECORD_SECONDS}초 동안 발음해주세요.")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    y = audio.flatten()

    feat = extract_audio_features(y, SAMPLE_RATE)
    if feat is None:
        print("  -> 녹음된 소리가 거의 없습니다. 마이크 볼륨/선택된 장치를 확인해주세요.")
        print("     (sd.query_devices() 로 장치 목록을 확인할 수 있습니다.)")
        return

    feature_keys = list(feat.keys())
    ensure_csv_header(feature_keys)
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), label] + [feat[k] for k in feature_keys])
    print(f"  -> 저장 완료. (평균 피치: {feat['mean_pitch']:.1f} Hz)")


def run_quick_classifier_check():
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:
        print("\n[안내] pandas / scikit-learn이 설치되어 있지 않아 분류기 테스트는 건너뜁니다.")
        print("      pip install pandas scikit-learn 로 설치 후 다시 실행하면")
        print("      자동으로 baseline 정확도를 계산해줍니다.")
        return

    if not os.path.exists(CSV_PATH):
        print("\n[안내] 저장된 데이터가 없어 분류기 테스트를 건너뜁니다.")
        return

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    label_counts = df["label"].value_counts()
    print("\n=== 라벨별 수집 샘플 수 ===")
    print(label_counts)

    if df["label"].nunique() < 2 or label_counts.min() < 3:
        print("\n[안내] 라벨 종류 또는 라벨당 샘플 수가 너무 적어 분류기 테스트를 생략합니다.")
        print("      최소 2개 이상의 라벨을, 각각 3회 이상 녹음해주세요.")
        return

    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    X = df[feature_cols].values
    y = df["label"].values

    n_splits = min(3, label_counts.min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf)

    print("\n=== 마이크(음향) 단독 baseline 분류 정확도 ===")
    print(f"  {n_splits}-fold 교차검증 평균 정확도: {scores.mean():.1%}  (각 fold: {[f'{s:.1%}' for s in scores]})")
    chance = 1.0 / df["label"].nunique()
    print(f"  참고: 라벨 {df['label'].nunique()}개 기준 우연 수준(chance) 정확도 = {chance:.1%}")
    print("\n  -> 6장에서 논의한 '음향 채널이 후방 조음(연구개음) 판별의 공백을")
    print("     일부 보완할 잠재력이 있다'는 가설을 실제 데이터로 확인해보기 위한")
    print("     참고용 baseline입니다.")


def main():
    print("=" * 60)
    print(" Deafness_AI - 02. 마이크(음향) 프로토타입")
    print("=" * 60)
    print("사용 가능한 라벨:")
    for k, v in LABEL_GUIDE.items():
        print(f"  {k:<3} : {v}")
    print("\n라벨을 입력하고 Enter를 누르면 녹음이 시작됩니다.")
    print("종료하려면 'q' 를 입력하세요.\n")

    try:
        while True:
            label = input("라벨 입력 (종료: q) > ").strip().lower()
            if label == "q":
                break
            if label not in LABEL_GUIDE:
                print(f"  알 수 없는 라벨입니다. 다음 중 하나를 입력하세요: {list(LABEL_GUIDE.keys())}")
                continue
            record_one_label(label)
    except KeyboardInterrupt:
        pass

    print(f"\n저장된 파일: {CSV_PATH}")
    run_quick_classifier_check()


if __name__ == "__main__":
    main()
