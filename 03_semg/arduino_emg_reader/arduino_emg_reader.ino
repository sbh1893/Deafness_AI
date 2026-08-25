/*
  arduino_emg_reader.ino

  목적
  ----
  Gravity Analog EMG Sensor(또는 유사 아날로그 EMG 센서) 1~2채널을
  가능한 한 일정한 샘플링 속도(목표 1000Hz)로 읽어서, PC로 시리얼
  전송하는 스케치입니다.

  왜 이렇게 만들었나 (A그룹 지식 반영)
  ----------------------------------
  - sEMG 유효 대역은 20~450Hz 정도이므로, 나이퀴스트 정리에 따라
    최소 900Hz 이상 샘플링이 필요합니다. 이 스케치는 micros()로
    시간을 직접 관리하여 1000Hz(1ms 간격)를 목표로 합니다.
  - 실제 달성되는 속도는 보드 성능·시리얼 설정에 따라 달라지므로,
    파이썬 쪽에서 실측 후 경고를 띄우도록 설계했습니다(아래 파이썬
    스크립트의 '실측 샘플링 레이트 확인' 부분 참고).
  - 시리얼 전송 바이트 수를 줄이기 위해 타임스탬프는 보내지 않고
    "ch0,ch1" 두 정수만 보냅니다. 대신 파이썬에서 목표 주기(1ms)를
    가정해 시간축을 재구성합니다.

  배선
  ----
  - EMG 센서 1 (예: 턱밑)  Signal → Arduino A0
  - EMG 센서 2 (예: 귀 아래, 선택) Signal → Arduino A1
  - 센서 2를 쓰지 않는다면 A1은 그대로 두어도 됩니다(0 근처 값이
    찍히지만 파이썬 쪽에서 2채널 모드를 껐다면 무시됩니다).

  중요: 보드 매니저에서 baud rate를 250000으로 설정한 이유는,
  1000Hz로 "ch0,ch1\n" (~9~10바이트)를 계속 보내려면
  115200bps로는 대역폭이 부족하기 때문입니다.
*/

const int CH0_PIN = A0;
const int CH1_PIN = A1;
const unsigned long TARGET_INTERVAL_US = 1000;   // 1000Hz 목표 (1ms)
const unsigned long BAUD_RATE = 250000;

unsigned long nextSampleTime = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  // 아두이노 우노/나노 기본 ADC 분해능(10비트, 0~1023)을 그대로 사용합니다.
  nextSampleTime = micros();
}

void loop() {
  unsigned long now = micros();

  // 목표 시각이 되기 전까지는 대기 (busy-wait 방식으로 지터를 최소화)
  if ((long)(now - nextSampleTime) < 0) {
    return;
  }

  int ch0 = analogRead(CH0_PIN);
  int ch1 = analogRead(CH1_PIN);

  Serial.print(ch0);
  Serial.print(',');
  Serial.println(ch1);

  // 다음 샘플 시각을 '현재 시각 + 간격'이 아니라 '이전 목표 시각 + 간격'으로
  // 잡아야 누적 지연(drift)이 생기지 않습니다.
  nextSampleTime += TARGET_INTERVAL_US;
}
