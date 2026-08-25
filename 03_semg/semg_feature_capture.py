# -*- coding: utf-8 -*-
"""
03_semg / semg_feature_capture.py

목적
----
아두이노(arduino_emg_reader.ino)에서 시리얼로 들어오는 EMG 원시 신호를
받아서, 실전 수준의 전처리(대역통과+노치 필터, 정류, 포락선, MVC 정규화)를
거친 뒤 라벨과 함께 저장하고, 세션 단위 교차검증으로 baseline 정확도를
확인하는 스크립트입니다.

01_webcam_vsr, 02_mic_acoustic 와 동일한 사용 흐름(라벨 입력 → 녹화 →
반복 → 'q' 종료 → 자동 baseline 확인)을 따릅니다.

시작하기 전에 꼭 확인하세요
--------------------------
1. 아두이노에 arduino_emg_reader.ino를 업로드했는지
2. 아래 SERIAL_PORT를 본인 환경에 맞게 수정했는지
   (Windows 예: "COM3", macOS/Linux 예: "/dev/tty.usbmodemXXXX")
3. 보드레이트가 아두이노 스케치와 동일(250000)한지
"""

import csv
import os
import time
import warnings
from datetime import datetime

import numpy as np
import serial
from scipy.signal import butter, filtfilt, iirnotch

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 설정값
# ----------------------------------------------------------------------
SERIAL_PORT = "COM3"          # Windows는 "COM3" 형태, macOS/Linux는 "/dev/tty.usbmodemXXXX" 형태
BAUD_RATE = 250000            # 아두이노 스케치와 반드시 동일해야 함
TARGET_FS = 1000.0            # 목표 샘플링 레이트(Hz). 아두이노 스케치의 TARGET_INTERVAL_US와 대응
TWO_CHANNEL_MODE = True       # False로 하면 ch0(턱밑)만 사용

RECORD_SECONDS = 1.5
MVC_SECONDS = 3.0             # 최대수의수축(MVC) 측정 시간
CSV_PATH = os.path.join(os.path.dirname(__file__), "semg_features.csv")

BANDPASS_LOW = 20.0            # sEMG 유효 대역 하한 (Hz) - 모션 아티팩트 제거
BANDPASS_HIGH = 450.0          # sEMG 유효 대역 상한 (Hz)
NOTCH_FREQ = 60.0              # 국내 전원 주파수(60Hz) 잡음 제거
ENVELOPE_CUTOFF = 6.0          # 포락선 추출용 저역통과 필터 컷오프 (Hz)

LABEL_GUIDE = {
    "m": "ㅁ (비음, 1순위)",
    "n": "ㄴ (비음, 1순위)",
    "ng": "ㅇ받침 (비음, 1순위)",
    "g": "ㄱ (연구개음, 2순위)",
    "k": "ㅋ (연구개음, 2순위)",
    "kk": "ㄲ (연구개음, 2순위)",
}


# ----------------------------------------------------------------------
# DSP 파이프라인 (A그룹 1번 지식 반영)
# ----------------------------------------------------------------------
def bandpass_filter(signal, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=4):
    """20~450Hz 대역통과 필터. sEMG 유효 대역만 남기고 저주파 모션
    아티팩트와 고주파 잡음을 제거한다."""
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)  # 나이퀴스트 한계를 넘지 않도록 보정
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, freq=NOTCH_FREQ, quality=30):
    """60Hz 전원 잡음 제거용 노치 필터."""
    nyq = fs / 2.0
    if freq >= nyq:
        return signal  # 샘플링 레이트가 너무 낮아 60Hz를 걸러낼 수 없는 경우
    b, a = iirnotch(freq / nyq, quality)
    return filtfilt(b, a, signal)


def rectify(signal):
    """정류: 신호를 절댓값으로 변환하여 '활동량'만 남긴다."""
    return np.abs(signal)


def extract_envelope(rectified_signal, fs, cutoff=ENVELOPE_CUTOFF, order=4):
    """포락선 추출: 정류된 신호에 저역통과 필터를 적용해 부드러운
    활동량 곡선을 얻는다."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, rectified_signal)


def preprocess_channel(raw, fs):
    """원시 신호 1채널에 대해 (대역통과 → 노치 → 정류 → 포락선) 전체
    파이프라인을 적용한다."""
    raw = np.asarray(raw, dtype=float)
    if len(raw) < 20:
        return None
    filtered = bandpass_filter(raw, fs)
    filtered = notch_filter(filtered, fs)
    rectified = rectify(filtered)
    envelope = extract_envelope(rectified, fs)
    return envelope


# ----------------------------------------------------------------------
# 시리얼 입출력
# ----------------------------------------------------------------------
def open_serial():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # 아두이노 리셋 대기
    ser.reset_input_buffer()
    return ser


def read_raw_samples(ser, duration_sec):
    """duration_sec 동안 시리얼에서 'ch0,ch1' 라인을 읽어 배열로 반환.
    실제 걸린 시간을 함께 반환하여 실측 샘플링 레이트 계산에 사용한다."""
    ser.reset_input_buffer()
    ch0_list, ch1_list = [], []
    start = time.time()
    while time.time() - start < duration_sec:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue
        try:
            ch0_list.append(int(parts[0]))
            ch1_list.append(int(parts[1]))
        except ValueError:
            continue
    elapsed = time.time() - start
    return np.array(ch0_list), np.array(ch1_list), elapsed


def check_actual_sampling_rate(n_samples, elapsed):
    """A그룹 3번 지식 반영: 이론상 가능한 샘플링 레이트와 실제
    달성된 레이트가 다를 수 있으므로 반드시 실측하고 경고한다."""
    if elapsed <= 0:
        return 0.0
    actual_fs = n_samples / elapsed
    ratio = actual_fs / TARGET_FS
    print(f"    [실측] 샘플링 레이트: {actual_fs:.0f} Hz (목표 {TARGET_FS:.0f} Hz 대비 {ratio:.0%})")
    if ratio < 0.8:
        print("    [경고] 목표 샘플링 레이트에 크게 못 미칩니다. USB 케이블/포트,")
        print("           보드레이트 설정, 다른 프로그램의 시리얼 점유 여부를 확인하세요.")
    return actual_fs


# ----------------------------------------------------------------------
# MVC(최대수의수축) 보정 (A그룹 1번 지식 반영)
# ----------------------------------------------------------------------
def calibrate_mvc(ser):
    print("\n=== MVC(최대수의수축) 보정 ===")
    print("사람마다, 세션마다 신호 크기가 다르기 때문에 보정이 필요합니다.")
    input(f"준비되면 Enter를 누른 뒤, {MVC_SECONDS:.0f}초 동안 해당 부위 근육에 최대한 힘을 줘보세요...")

    ch0_raw, ch1_raw, elapsed = read_raw_samples(ser, MVC_SECONDS)
    actual_fs = check_actual_sampling_rate(len(ch0_raw), elapsed)
    fs = actual_fs if actual_fs > 100 else TARGET_FS

    env0 = preprocess_channel(ch0_raw, fs)
    mvc0 = float(np.percentile(env0, 95)) if env0 is not None else 1.0

    mvc1 = 1.0
    if TWO_CHANNEL_MODE:
        env1 = preprocess_channel(ch1_raw, fs)
        mvc1 = float(np.percentile(env1, 95)) if env1 is not None else 1.0

    # 0으로 나누는 것을 방지
    mvc0 = max(mvc0, 1e-6)
    mvc1 = max(mvc1, 1e-6)

    print(f"    MVC(ch0) = {mvc0:.2f}" + (f", MVC(ch1) = {mvc1:.2f}" if TWO_CHANNEL_MODE else ""))
    return mvc0, mvc1, fs


# ----------------------------------------------------------------------
# CSV 저장
# ----------------------------------------------------------------------
def ensure_csv_header():
    is_new = not os.path.exists(CSV_PATH)
    if is_new:
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            header = ["timestamp", "session_id", "label",
                       "ch0_mean_norm", "ch0_rms_norm", "ch0_max_norm"]
            if TWO_CHANNEL_MODE:
                header += ["ch1_mean_norm", "ch1_rms_norm", "ch1_max_norm"]
            writer.writerow(header)


def record_one_label(ser, label, session_id, mvc0, mvc1, fs):
    print(f"  -> '{LABEL_GUIDE.get(label, label)}' 녹화 시작! {RECORD_SECONDS}초 동안 반복해서 발음해주세요.")
    ch0_raw, ch1_raw, elapsed = read_raw_samples(ser, RECORD_SECONDS)
    actual_fs = check_actual_sampling_rate(len(ch0_raw), elapsed)
    use_fs = actual_fs if actual_fs > 100 else fs

    env0 = preprocess_channel(ch0_raw, use_fs)
    if env0 is None:
        print("  -> 신호를 충분히 받지 못했습니다. 연결을 확인하고 다시 시도해주세요.")
        return

    # MVC 대비 정규화 (개인차·세션차 보정, A그룹 1번 지식 반영)
    env0_norm = env0 / mvc0
    row = [datetime.now().isoformat(), session_id, label,
           float(np.mean(env0_norm)), float(np.sqrt(np.mean(env0_norm ** 2))), float(np.max(env0_norm))]

    if TWO_CHANNEL_MODE:
        env1 = preprocess_channel(ch1_raw, use_fs)
        if env1 is not None:
            env1_norm = env1 / mvc1
            row += [float(np.mean(env1_norm)), float(np.sqrt(np.mean(env1_norm ** 2))), float(np.max(env1_norm))]
        else:
            row += [0.0, 0.0, 0.0]

    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"  -> 저장 완료. (ch0 평균 활동량: {row[3]:.2f} x MVC)")


# ----------------------------------------------------------------------
# 세션 독립적(session-independent) 교차검증 (A그룹 2번 지식 반영)
# ----------------------------------------------------------------------
def run_session_independent_check():
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GroupKFold, cross_val_score
    except ImportError:
        print("\n[안내] pandas / scikit-learn이 설치되어 있지 않아 분류기 테스트는 건너뜁니다.")
        return

    if not os.path.exists(CSV_PATH):
        print("\n[안내] 저장된 데이터가 없어 분류기 테스트를 건너뜁니다.")
        return

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    label_counts = df["label"].value_counts()
    session_counts = df["session_id"].nunique()

    print("\n=== 라벨별 수집 샘플 수 ===")
    print(label_counts)
    print(f"\n=== 세션 수: {session_counts} ===")

    if df["label"].nunique() < 2 or label_counts.min() < 3:
        print("\n[안내] 라벨 종류 또는 라벨당 샘플이 너무 적어 분류기 테스트를 생략합니다.")
        return

    feature_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["session_id"].values

    if session_counts < 2:
        print("\n[중요 안내] 세션이 1개뿐이라 '세션 독립적 검증'을 할 수 없습니다.")
        print("            지금 결과는 '같은 세션 안에서만 잘 맞히는지'를 본 것이며,")
        print("            8장에서 우려한 '세션 간 재현성'은 아직 확인되지 않았습니다.")
        print("            반드시 최소 2세션 이상 수집 후 다시 실행해주세요.")
        # 참고용으로 일반 K-fold 결과만 보여줌
        from sklearn.model_selection import StratifiedKFold
        n_splits = min(3, label_counts.min())
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(clf, X, y, cv=skf)
        print(f"\n  (참고, 세션 미분리) {n_splits}-fold 평균 정확도: {scores.mean():.1%}")
        return

    # 세션 단위로 완전히 분리하는 검증 - 이게 진짜 재현성 검증입니다.
    n_splits = min(session_counts, 5)
    gkf = GroupKFold(n_splits=n_splits)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, groups=groups, cv=gkf)

    print(f"\n=== 세션 독립적(session-independent) baseline 정확도 ===")
    print(f"  {n_splits}-fold(세션 단위 분리) 평균 정확도: {scores.mean():.1%}  (각 fold: {[f'{s:.1%}' for s in scores]})")
    chance = 1.0 / df["label"].nunique()
    print(f"  참고: 라벨 {df['label'].nunique()}개 기준 우연 수준(chance) 정확도 = {chance:.1%}")
    print("\n  -> 이 수치가 8장 파일럿 판정 기준(연구개음 세트 10~15%p 이상 개선)의")
    print("     실제 근거 데이터가 됩니다. 세션을 섞지 않고 검증했기 때문에,")
    print("     '새로운 날 다시 붙여도 통하는가'에 대한 현실적인 답입니다.")


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
def main():
    ensure_csv_header()

    print("=" * 60)
    print(" Deafness_AI - 03. sEMG 신호처리 & 분석")
    print("=" * 60)
    print(f"시리얼 포트: {SERIAL_PORT} @ {BAUD_RATE}bps  (2채널 모드: {TWO_CHANNEL_MODE})")

    try:
        ser = open_serial()
    except serial.SerialException as e:
        print(f"[오류] 시리얼 포트를 열 수 없습니다: {e}")
        print("       SERIAL_PORT 값을 본인 환경에 맞게 수정했는지 확인해주세요.")
        return

    session_id = input("\n세션 번호를 입력하세요 (예: 1, 2, 3 - 세션마다 전극을 다시 부착한 뒤 새 번호로 실행) > ").strip()

    mvc0, mvc1, fs = calibrate_mvc(ser)

    print("\n사용 가능한 라벨:")
    for k, v in LABEL_GUIDE.items():
        print(f"  {k:<3} : {v}")
    print("\n라벨을 입력하고 Enter를 누르면 녹화가 시작됩니다.")
    print("종료하려면 'q' 를 입력하세요.\n")

    try:
        while True:
            label = input("라벨 입력 (종료: q) > ").strip().lower()
            if label == "q":
                break
            if label not in LABEL_GUIDE:
                print(f"  알 수 없는 라벨입니다. 다음 중 하나를 입력하세요: {list(LABEL_GUIDE.keys())}")
                continue
            record_one_label(ser, label, session_id, mvc0, mvc1, fs)
    finally:
        ser.close()

    print(f"\n저장된 파일: {CSV_PATH}")
    run_session_independent_check()


if __name__ == "__main__":
    main()
