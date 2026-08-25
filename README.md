# Deafness_AI

청각장애인 자가 조음 훈련 시스템 MVP — 0-A단계 소프트웨어 사전 프로토타입

자세한 설명(작동 원리, 실행 방법, 결과 해석)은 [WIKI.md](./WIKI.md)를 확인하세요.

## 빠른 시작

```powershell
# 1. 웹캠 입모양 테스트
cd 01_webcam_vsr
pip install -r requirements.txt
python lip_landmark_capture.py

# 2. 마이크 음향 테스트
cd ../02_mic_acoustic
pip install -r requirements.txt
python audio_feature_capture.py

# 3. sEMG 테스트 (아두이노에 arduino_emg_reader.ino 업로드 먼저 필요)
cd ../03_semg
pip install -r requirements.txt
python semg_feature_capture.py   # SERIAL_PORT를 본인 환경에 맞게 수정 후 실행
```

01/02: 라벨(m/n/ng/g/k/kk) 입력 → 1.5초 녹화/녹음 → 반복 →
`q` 입력 시 baseline 정확도 자동 출력.

03(sEMG): 세션 번호 입력 → MVC 보정 → 라벨 입력 → 녹화 → 반복 →
`q` 입력 시 **세션 단위 분리** baseline 정확도 자동 출력 (최소 2세션 권장).
