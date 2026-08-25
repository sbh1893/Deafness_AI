# -*- coding: utf-8 -*-
"""
04_audio_to_tongue / 02_baseline_eigentongue.py

목적
----
가장 단순한 베이스라인: 오디오 특징(멜 스펙트로그램) 프레임 하나로
같은 시점의 조음 파라미터(eigentongue 계수, 01단계에서 만든 라벨)를
선형회귀로 예측한다.

이게 왜 첫 실험이어야 하는가
--------------------------
복잡한 딥러닝 모델을 만들기 전에, "오디오만으로 혀 위치를 얼마나
맞힐 수 있는지"의 하한선(lower bound)을 빠르게 확인하기 위함이다.
이 베이스라인보다 딥러닝 모델이 못하면, 애초에 딥러닝 쪽 설계가
잘못된 것이다. TaL 논문이 인용한 Fabre 외(2015)의 eigentongue +
얕은 신경망 방식과 같은 수준의 접근이다.
"""

import glob
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def load_split(split_name):
    """해당 split의 모든 npz 파일에서 (audio_frame, tongue_frame) 쌍을 모은다."""
    files = glob.glob(os.path.join(PROCESSED_DIR, split_name, "*.npz"))
    X, y, speakers = [], [], []
    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        audio = data["audio"]      # (T, N_MELS)
        tongue = data["tongue"]    # (T, TONGUE_PARAM_DIM)
        speaker = str(data["speaker"])
        n = min(len(audio), len(tongue))
        X.append(audio[:n])
        y.append(tongue[:n])
        speakers.extend([speaker] * n)
    if not X:
        return None, None, None
    return np.concatenate(X), np.concatenate(y), np.array(speakers)


def evaluate(model, X, y, name):
    pred = model.predict(X)
    print(f"\n=== {name} 성능 ===")
    for d in range(y.shape[1]):
        r, _ = pearsonr(y[:, d], pred[:, d])
        r2 = r2_score(y[:, d], pred[:, d])
        print(f"  조음 파라미터 {d+1}번 축: 상관계수 r={r:.3f}, R^2={r2:.3f}")

    # 참고용 baseline: "그냥 항상 평균값으로 예측"했을 때의 R^2 (항상 0.0)
    print(f"  참고: 아무 정보 없이 평균값만 예측하면 R^2=0.0 입니다.")
    print(f"        위 R^2가 0에 가깝다면, 오디오 정보가 이 축 예측에 거의 기여하지 못한다는 뜻입니다.")
    return pred


def main():
    print("데이터 로딩 중...")
    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, spk_test = load_split("test")

    if X_train is None:
        print("[안내] 학습 데이터가 없습니다. 먼저 01_prepare_data.py를 실행하세요.")
        return

    print(f"train 프레임 수: {len(X_train)}, val: {len(X_val) if X_val is not None else 0}, "
          f"test: {len(X_test) if X_test is not None else 0}")

    # 정규화 (화자마다 다른 오디오 특징 스케일 보정)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - mean) / std

    model = Ridge(alpha=1.0)
    model.fit(X_train_n, y_train)

    if X_val is not None:
        evaluate(model, (X_val - mean) / std, y_val, "검증셋(val, 학습에 없던 화자)")

    if X_test is not None:
        evaluate(model, (X_test - mean) / std, y_test, "테스트셋(test, 화자독립 최종 평가)")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump({"model": model, "mean": mean, "std": std},
                os.path.join(PROCESSED_DIR, "baseline_ridge.joblib"))
    print(f"\n베이스라인 모델 저장 완료: {os.path.join(PROCESSED_DIR, 'baseline_ridge.joblib')}")


if __name__ == "__main__":
    main()
