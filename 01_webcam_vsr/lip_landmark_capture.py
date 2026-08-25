# -*- coding: utf-8 -*-
"""
01_webcam_vsr / lip_landmark_capture.py

목적
----
웹캠만으로 입모양(입술 랜드마크) 기반 조음 인식이 얼마나 잘 되는지
사전 검증하기 위한 최소 프로토타입입니다. (하드웨어 추가 구매 없이 지금 바로 실행 가능)

동작 요약
--------
1. MediaPipe FaceMesh로 얼굴에서 입술 주요 랜드마크를 추출합니다.
2. 사용자가 콘솔에 목표 음소 라벨을 입력하면, 짧은 시간(RECORD_SECONDS) 동안
   웹캠 프레임을 연속 캡처하면서 입 모양 특징(너비, 높이, 세로/가로 비율 등)을
   프레임 단위로 CSV에 저장합니다.
3. 'q'를 입력해 종료하면, 지금까지 모은 데이터로 아주 간단한 분류기를 학습해
   "이 특징만으로 음소가 어느 정도 구분되는가"를 정확도 숫자로 보여줍니다.

자세한 설명은 상위 폴더의 WIKI.md 를 참고하세요.
"""

import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------------------------------------
# 설정값
# ----------------------------------------------------------------------
RECORD_SECONDS = 1.5          # 라벨 하나당 녹화 시간(초)
CSV_PATH = os.path.join(os.path.dirname(__file__), "lip_features.csv")

# 6장(목표 음소 범위 제안)에서 정한 1순위(비음) + 2순위(연구개음) 목표 음소
LABEL_GUIDE = {
    "m": "ㅁ (비음, 1순위)",
    "n": "ㄴ (비음, 1순위)",
    "ng": "ㅇ받침 (비음, 1순위)",
    "g": "ㄱ (연구개음, 2순위)",
    "k": "ㅋ (연구개음, 2순위)",
    "kk": "ㄲ (연구개음, 2순위)",
}

# MediaPipe FaceMesh에서 사용할 입술 관련 핵심 랜드마크 인덱스
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
FACE_LEFT = 234   # 얼굴 폭 정규화용(카메라와의 거리 편차 보정)
FACE_RIGHT = 454


def dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def extract_mouth_features(landmarks, image_w, image_h):
    """FaceMesh 랜드마크에서 입모양 관련 간단 기하 특징을 뽑는다.

    주의: 이것은 정식 CNN 기반 VSR이 아니라, '웹캠만으로 최소한의
    구분이 가능한가'를 빠르게 확인하기 위한 매우 단순화된 기하학적
    특징(너비/높이/비율)입니다. 본격 개발 단계에서는 CNN 기반 모델로
    교체하는 것을 전제로 합니다.
    """
    pts = [(lm.x * image_w, lm.y * image_h) for lm in landmarks]

    face_width = dist(pts[FACE_LEFT], pts[FACE_RIGHT])
    if face_width < 1e-6:
        return None

    mouth_width = dist(pts[MOUTH_LEFT], pts[MOUTH_RIGHT]) / face_width
    mouth_height = dist(pts[UPPER_LIP_TOP], pts[LOWER_LIP_BOTTOM]) / face_width
    aspect_ratio = mouth_height / mouth_width if mouth_width > 1e-6 else 0.0

    return {
        "mouth_width": mouth_width,
        "mouth_height": mouth_height,
        "aspect_ratio": aspect_ratio,
    }


def ensure_csv_header():
    is_new = not os.path.exists(CSV_PATH)
    if is_new:
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "label", "mouth_width", "mouth_height", "aspect_ratio"]
            )


def record_one_label(cap, face_mesh, label):
    print(f"  -> '{LABEL_GUIDE.get(label, label)}' 녹화 시작! {RECORD_SECONDS}초 동안 반복해서 발음해주세요.")
    time.sleep(0.5)  # 짧은 준비 시간

    rows = []
    start = time.time()
    while time.time() - start < RECORD_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            feat = extract_mouth_features(landmarks, w, h)
            if feat is not None:
                ts = datetime.now().isoformat()
                rows.append([ts, label, feat["mouth_width"], feat["mouth_height"], feat["aspect_ratio"]])

            # 시각 확인용: 입술 주요 점 표시
            for idx in (UPPER_LIP_TOP, LOWER_LIP_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT):
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        cv2.putText(frame, f"REC: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Deafness_AI - Webcam VSR Prototype", frame)
        cv2.waitKey(1)

    if rows:
        with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"  -> {len(rows)} 프레임 저장 완료.")
    else:
        print("  -> 얼굴/입 인식이 안 됐습니다. 카메라 각도를 확인하고 다시 시도해주세요.")


def run_quick_classifier_check():
    """모은 데이터로 아주 간단한 분류기를 학습해 baseline 정확도를 확인한다."""
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:
        print("\n[안내] pandas / scikit-learn이 설치되어 있지 않아 분류기 테스트는 건너뜁니다.")
        print("      pip install pandas scikit-learn 로 설치 후 이 스크립트를 다시 실행하면")
        print("      자동으로 baseline 정확도를 계산해줍니다.")
        return

    if not os.path.exists(CSV_PATH):
        print("\n[안내] 저장된 데이터가 없어 분류기 테스트를 건너뜁니다.")
        return

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    label_counts = df["label"].value_counts()
    print("\n=== 라벨별 수집 프레임 수 ===")
    print(label_counts)

    # 라벨이 2개 미만이거나, 각 라벨 샘플이 너무 적으면 스킵
    if df["label"].nunique() < 2 or label_counts.min() < 5:
        print("\n[안내] 라벨 종류 또는 라벨당 샘플 수가 너무 적어 분류기 테스트를 생략합니다.")
        print("      최소 2개 이상의 라벨을, 각각 5프레임 이상 모아주세요.")
        return

    X = df[["mouth_width", "mouth_height", "aspect_ratio"]].values
    y = df["label"].values

    n_splits = min(3, label_counts.min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf)

    print("\n=== 웹캠(입모양) 단독 baseline 분류 정확도 ===")
    print(f"  {n_splits}-fold 교차검증 평균 정확도: {scores.mean():.1%}  (각 fold: {[f'{s:.1%}' for s in scores]})")
    chance = 1.0 / df["label"].nunique()
    print(f"  참고: 라벨 {df['label'].nunique()}개 기준 우연 수준(chance) 정확도 = {chance:.1%}")
    print("\n  -> 이 수치는 4~5장에서 논의한 'VSR 단독으로는 연구개음 등 후방 조음이")
    print("     취약할 수 있다'는 가설을 실제 데이터로 확인해보기 위한 참고용 baseline입니다.")


def main():
    ensure_csv_header()

    print("=" * 60)
    print(" Deafness_AI - 01. 웹캠 입모양(VSR) 프로토타입")
    print("=" * 60)
    print("사용 가능한 라벨:")
    for k, v in LABEL_GUIDE.items():
        print(f"  {k:<3} : {v}")
    print("\n라벨을 입력하고 Enter를 누르면 촬영이 시작됩니다.")
    print("종료하려면 'q' 를 입력하세요.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Windows에서 기본 백엔드로 카메라가 안 열리면 아래 줄로 교체해보세요.
        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("[오류] 웹캠을 열 수 없습니다. 다른 프로그램이 카메라를 쓰고 있는지 확인해주세요.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as face_mesh:
        try:
            while True:
                label = input("라벨 입력 (종료: q) > ").strip().lower()
                if label == "q":
                    break
                if label not in LABEL_GUIDE:
                    print(f"  알 수 없는 라벨입니다. 다음 중 하나를 입력하세요: {list(LABEL_GUIDE.keys())}")
                    continue
                record_one_label(cap, face_mesh, label)
        finally:
            cap.release()
            cv2.destroyAllWindows()

    print(f"\n저장된 파일: {CSV_PATH}")
    run_quick_classifier_check()


if __name__ == "__main__":
    main()
