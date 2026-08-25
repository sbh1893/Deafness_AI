# -*- coding: utf-8 -*-
"""
04_audio_to_tongue / 01_prepare_data.py

목적
----
TaL Corpus(오디오 + 초음파 혀 영상 페어 데이터)를 읽어서:
1. 오디오 → 멜 스펙트로그램 특징 추출
2. 초음파 프레임 → PCA(eigentongue) 기반 저차원 조음 파라미터로 축소
3. 두 시계열의 시간축을 맞춤(정렬)
4. 화자 단위로 train/val/test를 나눠 저장

TaL 데이터 형식 (원 논문 기준)
------------------------------
- {utt}.wav   : 48kHz 16bit 오디오
- {utt}.ult   : 원시 초음파 데이터. 프레임당 64 scanline x 842 echo return
                (uint8), 즉 프레임 하나가 64*842 바이트.
- {utt}.param : 초음파 메타데이터(fps 등)를 담은 텍스트 파일(key=value 형식으로 가정)
- {utt}.txt   : 읽은 문장 텍스트

주의: .ult / .param 정확한 바이너리·필드 스펙은 배포처의 UltraSuite 툴킷
(https://github.com/UltraSuite/ultrasuite-tools) 문서를 통해 반드시
재확인하세요. 아래 파서는 논문에 명시된 스펙(64 scanline x 842 echo)을
기준으로 작성한 합리적 추정이며, 실제 파일에서 프레임 수가 안 맞으면
NUM_SCANLINES / NUM_ECHOES 값을 조정해야 할 수 있습니다.
"""

import json
import os
import random
from pathlib import Path

import librosa
import numpy as np
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# 설정값
# ----------------------------------------------------------------------
TAL_ROOT = os.environ.get("TAL_ROOT", "./TaL80")   # TaL 데이터 압축 해제 경로
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

NUM_SCANLINES = 64      # 초음파 스캔라인 수 (TaL 논문 기준)
NUM_ECHOES = 842        # 스캔라인당 echo return 수 (TaL 논문 기준)
ULTRASOUND_FPS_DEFAULT = 80.0  # .param에서 못 읽으면 쓰는 기본값 (논문 기준 ~80fps)

AUDIO_SR = 16000
N_MELS = 40
HOP_LENGTH = 160         # 16000/160 = 100fps, 오디오 특징 프레임레이트

TONGUE_PARAM_DIM = 3     # eigentongue 상위 몇 개 성분을 조음 파라미터로 쓸지
DOWNSAMPLE_ULT_SHAPE = (32, 84)  # PCA 전에 초음파 프레임을 이 크기로 축소(연산량 절감)

RANDOM_SEED = 42
VAL_SPEAKER_RATIO = 0.1
TEST_SPEAKER_RATIO = 0.1


# ----------------------------------------------------------------------
# 초음파 읽기
# ----------------------------------------------------------------------
def read_param_file(param_path):
    """key=value 또는 key:value 형식의 .param 파일을 최대한 유연하게 파싱한다."""
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
    """.ult 파일을 (T, num_scanlines, num_echoes) uint8 배열로 읽는다."""
    raw = np.fromfile(ult_path, dtype=np.uint8)
    frame_size = num_scanlines * num_echoes
    n_frames = raw.size // frame_size
    if n_frames == 0:
        raise ValueError(f"{ult_path}: 파일 크기가 예상 프레임 크기보다 작습니다. "
                          f"NUM_SCANLINES/NUM_ECHOES 설정을 확인하세요.")
    raw = raw[: n_frames * frame_size]
    frames = raw.reshape(n_frames, num_scanlines, num_echoes)
    return frames


def downsample_frames(frames, target_shape=DOWNSAMPLE_ULT_SHAPE):
    """PCA 연산량을 줄이기 위해 초음파 프레임을 간단히 블록 평균으로 축소한다."""
    t, h, w = frames.shape
    th, tw = target_shape
    h_bin, w_bin = h // th, w // tw
    if h_bin < 1 or w_bin < 1:
        return frames.astype(np.float32)
    trimmed = frames[:, : th * h_bin, : tw * w_bin]
    reshaped = trimmed.reshape(t, th, h_bin, tw, w_bin)
    return reshaped.mean(axis=(2, 4)).astype(np.float32)


# ----------------------------------------------------------------------
# 오디오 특징
# ----------------------------------------------------------------------
def extract_audio_features(wav_path, sr=AUDIO_SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    y, _ = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel).T  # (T, n_mels)
    return log_mel


# ----------------------------------------------------------------------
# 시간축 정렬
# ----------------------------------------------------------------------
def align_time_axes(audio_feats, tongue_params, ultrasound_fps, audio_fps):
    """오디오 프레임 수에 맞춰 조음 파라미터를 선형보간으로 리샘플링한다."""
    n_audio = audio_feats.shape[0]
    n_ult = tongue_params.shape[0]
    if n_ult < 2:
        return None

    audio_t = np.arange(n_audio) / audio_fps
    ult_t = np.arange(n_ult) / ultrasound_fps

    aligned = np.zeros((n_audio, tongue_params.shape[1]), dtype=np.float32)
    for d in range(tongue_params.shape[1]):
        aligned[:, d] = np.interp(audio_t, ult_t, tongue_params[:, d])
    return aligned


# ----------------------------------------------------------------------
# 화자 목록 찾기 & split
# ----------------------------------------------------------------------
def find_utterances(tal_root):
    """TaL80 디렉토리 구조를 순회하며 (speaker_id, utt_base_path) 목록을 만든다.

    실제 TaL 배포 디렉토리 구조는 화자별 폴더로 되어 있을 가능성이 높으므로,
    이 함수는 '화자 폴더 하위에 {utt}.wav/.ult/.param/.txt가 있다'는 가정으로
    작성되었습니다. 실제 압축 해제 후 구조가 다르면 이 부분만 수정하면 됩니다.
    """
    utterances = []
    root = Path(tal_root)
    if not root.exists():
        return utterances
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        for wav_path in speaker_dir.glob("*aud*.wav"):
            # 무음(sil)/속삭임(whi) 발화는 제외하고, 발성(aud) 발화만 사용
            base = wav_path.with_suffix("")
            ult_path = base.with_suffix(".ult")
            param_path = base.with_suffix(".param")
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
    """전체 발화 중 일부를 샘플링해 PCA(eigentongue)를 학습한다."""
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
        flat = small.reshape(small.shape[0], -1)
        sample_vectors.append(flat)

    if not sample_vectors:
        raise RuntimeError("PCA 학습용 초음파 프레임을 하나도 읽지 못했습니다. "
                            "TAL_ROOT 경로와 데이터 구조를 확인하세요.")

    all_vectors = np.concatenate(sample_vectors, axis=0)
    pca = PCA(n_components=TONGUE_PARAM_DIM, random_state=RANDOM_SEED)
    pca.fit(all_vectors)
    print(f"  PCA 학습 완료. 설명된 분산 비율: {pca.explained_variance_ratio_.sum():.1%}")
    return pca


def process_utterance(speaker_id, base, pca, split_name, out_dir):
    wav_path = base + ".wav"
    ult_path = base + ".ult"
    param_path = base + ".param"

    audio_feats = extract_audio_features(wav_path)
    frames = read_ultrasound_raw(ult_path)
    small = downsample_frames(frames)
    flat = small.reshape(small.shape[0], -1)
    tongue_params = pca.transform(flat)  # (T_ult, TONGUE_PARAM_DIM)

    ult_fps = get_ultrasound_fps(param_path)
    audio_fps = AUDIO_SR / HOP_LENGTH

    aligned_tongue = align_time_axes(audio_feats, tongue_params, ult_fps, audio_fps)
    if aligned_tongue is None:
        return False

    utt_name = os.path.basename(base)
    out_path = os.path.join(out_dir, split_name, f"{speaker_id}__{utt_name}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, audio=audio_feats.astype(np.float32),
              tongue=aligned_tongue.astype(np.float32), speaker=speaker_id)
    return True


def main():
    utterances = find_utterances(TAL_ROOT)
    print(f"총 {len(utterances)}개 발화를 찾았습니다. (TAL_ROOT={TAL_ROOT})")
    if not utterances:
        print("[안내] 데이터가 없어 종료합니다. TAL_ROOT 환경변수를 TaL80 압축 해제 경로로 지정하세요.")
        print("       예: set TAL_ROOT=D:\\datasets\\TaL80  (Windows)")
        return

    pca = fit_pca_on_sample(utterances)

    speaker_ids = [s for s, _ in utterances]
    splits = split_speakers(speaker_ids)
    print(f"화자 분할 — train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    print("[2/3] 발화별 특징 추출 및 정렬 중...")
    for speaker_id, base in utterances:
        split_name = next((s for s in ("train", "val", "test") if speaker_id in splits[s]), "train")
        try:
            ok = process_utterance(speaker_id, base, pca, split_name, OUTPUT_DIR)
            if ok:
                counts[split_name] += 1
        except Exception as e:
            print(f"  [실패] {base}: {e}")

    print(f"[3/3] 완료. 저장된 발화 수: {counts}")

    # PCA 모델과 split 정보를 함께 저장 (이후 단계에서 재사용)
    import joblib
    joblib.dump(pca, os.path.join(OUTPUT_DIR, "eigentongue_pca.joblib"))
    with open(os.path.join(OUTPUT_DIR, "splits.json"), "w") as f:
        json.dump({k: sorted(v) for k, v in splits.items()}, f, ensure_ascii=False, indent=2)
    print(f"결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
