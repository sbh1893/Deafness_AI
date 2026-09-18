# -*- coding: utf-8 -*-
"""
04_audio_to_tongue / 01_prepare_data.py  (오디오+입술영상 버전)

목적
----
TaL Corpus(오디오 + 초음파 혀 영상 + 입술 영상 페어 데이터)를 읽어서:
1. 오디오 → 멜 스펙트로그램 특징 추출
2. 입술 영상(.mp4) → MediaPipe로 입모양 기하학적 특징(너비/높이/비율) 추출
   (01_webcam_vsr과 동일한 방식 — "오디오만 vs 오디오+입모양" 비교를
   공정하게 하기 위해 실사용 때와 같은 특징을 씁니다)
3. 초음파 프레임 → PCA(eigentongue) 기반 저차원 조음 파라미터로 축소
4. 세 시계열의 시간축을 모두 오디오 프레임 기준으로 정렬
5. 화자 단위로 train/val/test를 나눠 저장 (오디오, 입모양, 혀 위치를 한 파일에 함께)

이번 버전에서 추가된 것
----------------------
- extract_video_features(): .mp4에서 프레임별 입모양 특징 추출
- 얼굴이 인식되지 않는 프레임은 직전 값으로 채움(forward-fill), 앞부분에
  인식 실패가 이어지면 0으로 채움 — 완전히 건너뛰지 않고 시계열 길이를
  유지하기 위함
- process_utterance()가 .mp4가 없는 발화는 건너뜁니다 — "오디오만" 실험과
  "오디오+입모양" 실험을 정확히 같은 발화 집합으로 공정 비교하기 위함

TaL 데이터 형식 (원 논문 기준)
------------------------------
- {utt}.wav   : 48kHz 16bit 오디오
- {utt}.mp4   : 입술 영상 (오디오와 동기화, 논문 기준 약 60fps)
- {utt}.ult   : 원시 초음파 데이터. 프레임당 64 scanline x 842 echo return
- {utt}.param : 초음파 메타데이터(fps 등)
- {utt}.txt   : 읽은 문장 텍스트

주의: .ult/.param, 그리고 영상 fps는 논문에 기술된 스펙 기반 추정입니다.
실제 데이터로 처음 실행할 때 소량으로 반드시 검증하세요.
"""

import json
import os
import random
from pathlib import Path

import cv2
import librosa
import mediapipe as mp
import numpy as np
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# 설정값
# ----------------------------------------------------------------------
TAL_ROOT = os.environ.get("TAL_ROOT", "./TaL80")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

NUM_SCANLINES = 64
NUM_ECHOES = 842
ULTRASOUND_FPS_DEFAULT = 80.0
VIDEO_FPS_DEFAULT = 60.0   # TaL 논문 기준 입술 영상 프레임레이트 추정치

AUDIO_SR = 16000
N_MELS = 40
HOP_LENGTH = 160  # 16000/160 = 100fps

TONGUE_PARAM_DIM = 3
DOWNSAMPLE_ULT_SHAPE = (32, 84)
VIDEO_FEATURE_DIM = 3  # mouth_width, mouth_height, aspect_ratio (01_webcam_vsr과 동일)

RANDOM_SEED = 42
VAL_SPEAKER_RATIO = 0.1
TEST_SPEAKER_RATIO = 0.1

# 01_webcam_vsr과 동일한 MediaPipe 랜드마크 인덱스 (일관성 유지가 핵심)
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
FACE_LEFT = 234
FACE_RIGHT = 454


# ----------------------------------------------------------------------
# 초음파 읽기 (변경 없음)
# ----------------------------------------------------------------------
def read_param_file(param_path):
    params = {}
    if not os.path.exists(param_path):
        return params
    with open(param_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("=", ":"):
                if sep in line:
                    k, v = line.split(sep, 1)
                    params[k.strip()] = v.strip()
                    break
    return params


def get_ultrasound_fps(param_path):
    params = read_param_file(param_path)
    for key in ("Fps", "fps", "FramesPerSec", "FrameRate"):
        if key in params:
            try:
                return float(params[key])
            except ValueError:
                pass
    return ULTRASOUND_FPS_DEFAULT


def read_ultrasound_raw(ult_path, num_scanlines=NUM_SCANLINES, num_echoes=NUM_ECHOES):
    raw = np.fromfile(ult_path, dtype=np.uint8)
    frame_size = num_scanlines * num_echoes
    n_frames = raw.size // frame_size
    if n_frames == 0:
        raise ValueError(f"{ult_path}: 파일 크기가 예상 프레임 크기보다 작습니다.")
    raw = raw[: n_frames * frame_size]
    return raw.reshape(n_frames, num_scanlines, num_echoes)


def downsample_frames(frames, target_shape=DOWNSAMPLE_ULT_SHAPE):
    t, h, w = frames.shape
    th, tw = target_shape
    h_bin, w_bin = h // th, w // tw
    if h_bin < 1 or w_bin < 1:
        return frames.astype(np.float32)
    trimmed = frames[:, : th * h_bin, : tw * w_bin]
    reshaped = trimmed.reshape(t, th, h_bin, tw, w_bin)
    return reshaped.mean(axis=(2, 4)).astype(np.float32)


# ----------------------------------------------------------------------
# 오디오 특징 (변경 없음)
# ----------------------------------------------------------------------
def extract_audio_features(wav_path, sr=AUDIO_SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    y, _ = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    return librosa.power_to_db(mel).T


# ----------------------------------------------------------------------
# 입술 영상 특징 (신규)
# ----------------------------------------------------------------------
_face_mesh_singleton = None


def _get_face_mesh():
    """FaceMesh 인스턴스를 재사용한다 (발화마다 새로 만들면 느림)."""
    global _face_mesh_singleton
    if _face_mesh_singleton is None:
        _face_mesh_singleton = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
    return _face_mesh_singleton


def _dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def extract_video_features(mp4_path):
    """01_webcam_vsr과 동일한 3개 특징(너비/높이/비율)을 영상 전체 프레임에서 뽑는다.
    얼굴 인식 실패 프레임은 forward-fill로 채운다."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return None

    face_mesh = _get_face_mesh()
    feats = []
    last_valid = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            pts = [(p.x * w, p.y * h) for p in lm]
            face_width = _dist(pts[FACE_LEFT], pts[FACE_RIGHT])
            if face_width > 1e-6:
                mouth_width = _dist(pts[MOUTH_LEFT], pts[MOUTH_RIGHT]) / face_width
                mouth_height = _dist(pts[UPPER_LIP_TOP], pts[LOWER_LIP_BOTTOM]) / face_width
                aspect = mouth_height / mouth_width if mouth_width > 1e-6 else 0.0
                last_valid = np.array([mouth_width, mouth_height, aspect], dtype=np.float32)

        feats.append(last_valid if last_valid is not None else np.zeros(VIDEO_FEATURE_DIM, dtype=np.float32))

    cap.release()
    if not feats:
        return None
    return np.stack(feats, axis=0)  # (T_video, 3)


# ----------------------------------------------------------------------
# 시간축 정렬 (범용화: 혀 위치·입모양 둘 다 이 함수로 처리)
# ----------------------------------------------------------------------
def align_to_audio(audio_feats, series, series_fps, audio_fps):
    """series(초음파 조음 파라미터 또는 입모양 특징)를 오디오 프레임 수에 맞춰
    선형보간으로 리샘플링한다."""
    n_audio = audio_feats.shape[0]
    n_series = series.shape[0]
    if n_series < 2:
        return None

    audio_t = np.arange(n_audio) / audio_fps
    series_t = np.arange(n_series) / series_fps

    aligned = np.zeros((n_audio, series.shape[1]), dtype=np.float32)
    for d in range(series.shape[1]):
        aligned[:, d] = np.interp(audio_t, series_t, series[:, d])
    return aligned


# ----------------------------------------------------------------------
# 화자 목록 찾기 & split (변경 없음)
# ----------------------------------------------------------------------
def find_utterances(tal_root):
    utterances = []
    root = Path(tal_root)
    if not root.exists():
        return utterances
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        for wav_path in speaker_dir.glob("*aud*.wav"):
            base = wav_path.with_suffix("")
            ult_path = base.with_suffix(".ult")
            if ult_path.exists():
                utterances.append((speaker_id, str(base)))
    return utterances


def split_speakers(speaker_ids, seed=RANDOM_SEED):
    speakers = sorted(set(speaker_ids))
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n = len(speakers)
    n_val = max(1, int(n * VAL_SPEAKER_RATIO))
    n_test = max(1, int(n * TEST_SPEAKER_RATIO))
    test_speakers = set(speakers[:n_test])
    val_speakers = set(speakers[n_test:n_test + n_val])
    train_speakers = set(speakers[n_test + n_val:])
    return {"train": train_speakers, "val": val_speakers, "test": test_speakers}


# ----------------------------------------------------------------------
# 메인 파이프라인
# ----------------------------------------------------------------------
def fit_pca_on_sample(utterances, max_frames_for_pca=20000):
    print("[1/3] PCA(eigentongue) 학습을 위한 초음파 프레임 샘플링 중...")
    sample_vectors = []
    rng = random.Random(RANDOM_SEED)
    shuffled = utterances[:]
    rng.shuffle(shuffled)

    for speaker_id, base in shuffled:
        if len(sample_vectors) >= max_frames_for_pca:
            break
        try:
            frames = read_ultrasound_raw(base + ".ult")
        except Exception as e:
            print(f"  [건너뜀] {base}: {e}")
            continue
        small = downsample_frames(frames)
        sample_vectors.append(small.reshape(small.shape[0], -1))

    if not sample_vectors:
        raise RuntimeError("PCA 학습용 초음파 프레임을 하나도 읽지 못했습니다.")

    all_vectors = np.concatenate(sample_vectors, axis=0)
    pca = PCA(n_components=TONGUE_PARAM_DIM, random_state=RANDOM_SEED)
    pca.fit(all_vectors)
    print(f"  PCA 학습 완료. 설명된 분산 비율: {pca.explained_variance_ratio_.sum():.1%}")
    return pca


def process_utterance(speaker_id, base, pca, split_name, out_dir):
    wav_path = base + ".wav"
    ult_path = base + ".ult"
    param_path = base + ".param"
    mp4_path = base + ".mp4"

    # "오디오만" vs "오디오+입모양"을 공정 비교하려면 .mp4가 없는 발화는 아예 제외
    if not os.path.exists(mp4_path):
        return False, "no_video"

    audio_feats = extract_audio_features(wav_path)

    frames = read_ultrasound_raw(ult_path)
    small = downsample_frames(frames)
    tongue_params = pca.transform(small.reshape(small.shape[0], -1))
    ult_fps = get_ultrasound_fps(param_path)
    audio_fps = AUDIO_SR / HOP_LENGTH
    aligned_tongue = align_to_audio(audio_feats, tongue_params, ult_fps, audio_fps)

    video_feats = extract_video_features(mp4_path)
    if video_feats is None:
        return False, "video_read_fail"
    aligned_video = align_to_audio(audio_feats, video_feats, VIDEO_FPS_DEFAULT, audio_fps)

    if aligned_tongue is None or aligned_video is None:
        return False, "align_fail"

    utt_name = os.path.basename(base)
    out_path = os.path.join(out_dir, split_name, f"{speaker_id}__{utt_name}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
              audio=audio_feats.astype(np.float32),
              video=aligned_video.astype(np.float32),
              tongue=aligned_tongue.astype(np.float32),
              speaker=speaker_id)
    return True, "ok"


def main():
    utterances = find_utterances(TAL_ROOT)
    print(f"총 {len(utterances)}개 발화를 찾았습니다. (TAL_ROOT={TAL_ROOT})")
    if not utterances:
        print("[안내] 데이터가 없어 종료합니다. TAL_ROOT 환경변수를 확인하세요.")
        return

    pca = fit_pca_on_sample(utterances)

    speaker_ids = [s for s, _ in utterances]
    splits = split_speakers(speaker_ids)
    print(f"화자 분할 — train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    skip_reasons = {}
    print("[2/3] 발화별 특징 추출 및 정렬 중 (오디오+입모양+혀위치)...")
    for speaker_id, base in utterances:
        split_name = next((s for s in ("train", "val", "test") if speaker_id in splits[s]), "train")
        try:
            ok, reason = process_utterance(speaker_id, base, pca, split_name, OUTPUT_DIR)
            if ok:
                counts[split_name] += 1
            else:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        except Exception as e:
            print(f"  [실패] {base}: {e}")
            skip_reasons["exception"] = skip_reasons.get("exception", 0) + 1

    print(f"[3/3] 완료. 저장된 발화 수: {counts}")
    if skip_reasons:
        print(f"       건너뛴 발화 사유: {skip_reasons}")

    import joblib
    joblib.dump(pca, os.path.join(OUTPUT_DIR, "eigentongue_pca.joblib"))
    with open(os.path.join(OUTPUT_DIR, "splits.json"), "w") as f:
        json.dump({k: sorted(v) for k, v in splits.items()}, f, ensure_ascii=False, indent=2)
    print(f"결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
