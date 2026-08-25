# -*- coding: utf-8 -*-
"""
04_audio_to_tongue / 05_realtime_inference.py

목적
----
학습된 CNN 모델(03단계 산출물)을 이용해, 실시간 마이크 입력으로부터
혀 위치 파라미터를 추정하고 콘솔에 간단히 시각화하는 데모.

이 스크립트는 화면 UI(웹캠 입모양 비교 + 혀 위치 참조 단면도)를
그대로 구현한 것은 아니고, "모델이 실시간으로 잘 도는지"를 빠르게
확인하기 위한 콘솔 버전입니다. 실제 앱에서는 이 추론 결과를
01_webcam_vsr의 입모양 결과와 함께 화면에 그리면 됩니다.
"""

import os
import time

import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import librosa

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_SR = 16000
N_MELS = 40
HOP_LENGTH = 160
WINDOW_SECONDS = 1.0     # 추론 1회에 사용할 오디오 길이
UPDATE_INTERVAL = 0.5    # 추론 주기(초)

AXIS_NAMES = ["혀 전후 위치", "혀 고저 위치", "3번 축(해석 필요)"]


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


def load_model():
    ckpt_path = os.path.join(PROCESSED_DIR, "cnn_model_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("cnn_model_best.pt가 없습니다. 03_train_model.py를 먼저 실행하세요.")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = AudioToTongueCNN(ckpt["n_mels"], ckpt["tongue_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def extract_features(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=AUDIO_SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel).T
    return log_mel.astype(np.float32)


def render_bar(value, axis_range=(-50, 50), width=20):
    """-축범위~+축범위 값을 [-------|----] 형태의 텍스트 막대로 표시한다."""
    lo, hi = axis_range
    ratio = (value - lo) / (hi - lo)
    ratio = max(0.0, min(1.0, ratio))
    pos = int(ratio * width)
    bar = ["-"] * width
    bar[pos] = "●"
    return "[" + "".join(bar) + f"]  {value:+.1f}"


def main():
    print("=" * 60)
    print(" Deafness_AI - 04. 오디오 기반 혀 위치 실시간 추론 데모")
    print("=" * 60)
    model = load_model()

    buffer = np.zeros(int(AUDIO_SR * WINDOW_SECONDS), dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal buffer
        mono = indata[:, 0]
        buffer = np.roll(buffer, -len(mono))
        buffer[-len(mono):] = mono

    print("마이크 스트리밍을 시작합니다. Ctrl+C로 종료하세요.\n")
    with sd.InputStream(samplerate=AUDIO_SR, channels=1, callback=callback):
        try:
            while True:
                time.sleep(UPDATE_INTERVAL)
                feats = extract_features(buffer.copy())
                x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred = model(x).cpu().numpy()[0]
                # 최근 프레임(윈도우 끝부분) 예측값을 대표값으로 사용
                latest = pred[-5:].mean(axis=0)

                print("\033c", end="")  # 콘솔 화면 지우기 (터미널에 따라 동작 안 할 수 있음)
                print("=== 실시간 혀 위치 추정 (참고용, 개략적 수치) ===")
                for i, name in enumerate(AXIS_NAMES[:len(latest)]):
                    print(f"{name:12s} {render_bar(latest[i])}")
                print("\n(Ctrl+C로 종료)")
        except KeyboardInterrupt:
            print("\n종료합니다.")


if __name__ == "__main__":
    main()
