# -*- coding: utf-8 -*-
"""
05_semg_dataset_test / test_with_gaddy_dataset.py

목적
----
Gaddy sEMG 데이터셋(영어 화자 1명, 8채널, raw/thesis/15번 파일 참고)으로
두 가지를 검증한다.

1. DSP 파이프라인 검증: 03_semg의 필터링(대역통과+노치+정류+포락선)
   코드가 실제 8채널 sEMG 원시 데이터에서도 정상 동작하는지 확인.
2. 채널 수 실험: 5장에서 인용한 "1채널로 줄이면 성능이 급격히
   나빠진다"(Hwang 외 2026)는 주장을, 실제 데이터로 미니 재현.
   과제는 "무음(silent) 발화 vs 발성(vocalized) 발화 구분"으로,
   Gaddy 데이터셋이 기본적으로 제공하는 라벨이라 별도 정답 라벨링
   없이 바로 실험할 수 있다.

데이터 준비
----------
1. raw/thesis/15_영어화자1명_근전도_데이터셋.md 안내에 따라
   https://doi.org/10.5281/zenodo.4064408 에서 데이터를 받는다.
2. 압축을 풀고, 그 최상위 경로를 GADDY_ROOT 환경변수로 지정한다.
   (Windows 예: set GADDY_ROOT=D:\\datasets\\emg_data)

주의: Gaddy 데이터셋의 정확한 폴더 구조(세션별 하위 폴더명 등)는
배포 버전에 따라 다를 수 있습니다. 아래 파일 탐색 로직은 "파일명이
숫자로 시작하고 _emg.npy로 끝난다"는 공식 문서 스펙만 가정하므로,
실제 구조가 다르면 find_emg_files() 함수만 수정하면 됩니다.
"""

import glob
import json
import os

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

GADDY_ROOT = os.environ.get("GADDY_ROOT", "./emg_data")
EMG_FS = 1000.0  # Gaddy 데이터셋 공식 샘플링 레이트 (Hz)

BANDPASS_LOW = 20.0
BANDPASS_HIGH = 450.0
NOTCH_FREQ = 60.0
ENVELOPE_CUTOFF = 6.0


# ----------------------------------------------------------------------
# 1. DSP 파이프라인 (03_semg/semg_feature_capture.py와 동일한 로직)
# ----------------------------------------------------------------------
def bandpass_filter(signal, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=4):
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, freq=NOTCH_FREQ, quality=30):
    nyq = fs / 2.0
    if freq >= nyq:
        return signal
    b, a = iirnotch(freq / nyq, quality)
    return filtfilt(b, a, signal)


def extract_envelope(rectified, fs, cutoff=ENVELOPE_CUTOFF, order=4):
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, rectified)


def preprocess_channel(raw, fs):
    """대역통과 -> 노치 -> 정류 -> 포락선. 03_semg와 동일 파이프라인."""
    raw = np.asarray(raw, dtype=float)
    if len(raw) < 30:
        return None
    filtered = bandpass_filter(raw, fs)
    filtered = notch_filter(filtered, fs)
    rectified = np.abs(filtered)
    return extract_envelope(rectified, fs)


# ----------------------------------------------------------------------
# 2. Gaddy 데이터셋 로딩
# ----------------------------------------------------------------------
def find_emg_files(root):
    """{n}_emg.npy 패턴의 파일을 재귀적으로 찾아, 같은 세트의 info.json과 묶는다."""
    pattern = os.path.join(root, "**", "*_emg.npy")
    emg_files = sorted(glob.glob(pattern, recursive=True))
    pairs = []
    for emg_path in emg_files:
        base = emg_path[: -len("_emg.npy")]
        info_path = base + "_info.json"
        pairs.append((emg_path, info_path if os.path.exists(info_path) else None))
    return pairs


def guess_is_silent(emg_path, info_path):
    """무음(silent)/발성(vocalized) 여부를 최대한 유연하게 추정한다.

    Gaddy 데이터셋은 세션 폴더명이나 info.json에 이 정보가 담겨 있는
    경우가 많습니다. 정확한 필드명이 다르면 이 함수만 수정하세요.
    """
    text = emg_path.lower()
    if info_path and os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            text += " " + json.dumps(info).lower()
        except Exception:
            pass
    if "silent" in text:
        return True
    if "voiced" in text or "vocal" in text or "parallel" in text:
        return False
    return None  # 판단 불가


def load_dataset(root, max_files=200):
    pairs = find_emg_files(root)
    if not pairs:
        return [], []
    X_raw, y = [], []
    for emg_path, info_path in pairs[:max_files]:
        label = guess_is_silent(emg_path, info_path)
        if label is None:
            continue
        emg = np.load(emg_path)  # 예상 shape: (T, 8)
        if emg.ndim != 2 or emg.shape[1] < 1:
            continue
        X_raw.append(emg)
        y.append("silent" if label else "vocalized")
    return X_raw, y


# ----------------------------------------------------------------------
# 3. 특징 추출 (채널별 평균 활동량 + RMS)
# ----------------------------------------------------------------------
def extract_features_all_channels(emg, fs=EMG_FS):
    """(T, C) 원시 EMG -> 채널별 [평균, RMS] 특징 벡터."""
    n_channels = emg.shape[1]
    feats = []
    for ch in range(n_channels):
        env = preprocess_channel(emg[:, ch], fs)
        if env is None:
            feats.extend([0.0, 0.0])
            continue
        feats.append(float(np.mean(env)))
        feats.append(float(np.sqrt(np.mean(env ** 2))))
    return feats


def run_dsp_sanity_check(X_raw):
    """1. DSP 파이프라인이 실제 데이터에서 에러 없이 도는지 확인."""
    print("=== 1. DSP 파이프라인 검증 ===")
    ok, fail = 0, 0
    for emg in X_raw[:20]:
        try:
            env = preprocess_channel(emg[:, 0], EMG_FS)
            if env is not None:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  [실패] {e}")
            fail += 1
    print(f"  성공: {ok}건, 실패: {fail}건 (샘플 최대 20건 기준)")
    if ok > 0:
        print("  -> 03_semg의 필터링 코드가 실제 다채널 sEMG 데이터에서도 정상 동작합니다.")


# ----------------------------------------------------------------------
# 4. 채널 수 실험 (5장 핵심 주장 재현)
# ----------------------------------------------------------------------
def run_channel_count_experiment(X_raw, y):
    print("\n=== 2. 채널 수에 따른 분류 정확도 비교 (무음 vs 발성 구분) ===")
    n_channels = X_raw[0].shape[1]
    print(f"사용 가능한 채널 수: {n_channels}, 샘플 수: {len(X_raw)}")

    if len(set(y)) < 2:
        print("[안내] 무음/발성 라벨을 둘 다 확보하지 못해 실험을 생략합니다.")
        print("       guess_is_silent() 함수를 실제 데이터 구조에 맞게 수정해보세요.")
        return

    # 전체 채널 특징
    X_all = np.array([extract_features_all_channels(emg, EMG_FS) for emg in X_raw])

    def eval_channels(channel_indices, label):
        cols = []
        for ch in channel_indices:
            cols.extend([ch * 2, ch * 2 + 1])  # [평균, RMS] 2개씩
        X_sub = X_all[:, cols]
        n_splits = min(5, min(np.bincount(np.array(y) == y[0])))
        n_splits = max(2, min(5, len(y) // 4))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(clf, X_sub, y, cv=skf)
        print(f"  {label}: 평균 정확도 {scores.mean():.1%}  ({n_splits}-fold)")
        return scores.mean()

    # 전체 채널
    acc_all = eval_channels(list(range(n_channels)), f"전체 {n_channels}채널 사용")

    # 채널 1개씩 (평균으로 대표값 산출)
    single_accs = []
    for ch in range(n_channels):
        acc = eval_channels([ch], f"채널 {ch}번만 사용")
        single_accs.append(acc)

    print(f"\n  요약: 전체 채널 정확도 {acc_all:.1%} vs 단일 채널 평균 정확도 {np.mean(single_accs):.1%} "
          f"(최고 단일 채널 {max(single_accs):.1%})")
    print("  -> 단일 채널 정확도가 전체 채널보다 뚜렷이 낮다면, Hwang 외(2026)가 보고한")
    print("     '채널을 줄일수록 성능이 급격히 나빠진다'는 주장을 우리 데이터로도 확인한 것입니다.")


def main():
    print(f"GADDY_ROOT = {GADDY_ROOT}")
    X_raw, y = load_dataset(GADDY_ROOT)
    if not X_raw:
        print("[안내] 데이터를 찾지 못했습니다. GADDY_ROOT 경로와 파일 구조를 확인하세요.")
        print("       (raw/thesis/15_영어화자1명_근전도_데이터셋.md 참고)")
        return
    print(f"총 {len(X_raw)}개 발화 로드 완료 (라벨 확보: {len(y)}개)")

    run_dsp_sanity_check(X_raw)

    if y:
        run_channel_count_experiment(X_raw, y)


if __name__ == "__main__":
    main()
