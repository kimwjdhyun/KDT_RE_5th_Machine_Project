#include <Wire.h>
#include <Adafruit_NeoPixel.h>
#include <string.h>

#define LED_PIN       7
#define PUMP_PIN      9
#define NUMPIXELS     60
#define SLAVE_ADDR    8

#define PUMP_DURATION 1000
#define PUMP_INTERVAL 86400000UL   // 하루 1회

struct DeviceState {
  uint8_t pumpState;
  uint8_t ledState;
};

Adafruit_NeoPixel strip(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

volatile uint8_t energyMode = 2;
volatile bool waterAlert = false;
volatile bool pumpRunning = false;
volatile bool ledOn = false;

DeviceState stateToSend;

unsigned long lastPumpTime = 0;
bool printFlag = false;

// ----------------------------
// LED 제어
// mode 0: OFF
// mode 1: 저전력
// mode 2: 정상
// ----------------------------
void controlLED(uint8_t mode) {
  uint8_t brightness = 0;

  if (mode == 0) {
    brightness = 0;
  } else if (mode == 1) {
    brightness = 40;
  } else {
    brightness = 100;
  }

  strip.setBrightness(brightness);

  for (int i = 0; i < NUMPIXELS; i++) {
    strip.setPixelColor(i, 255, 0, 255);
  }

  strip.show();
  ledOn = (brightness > 0);
}

// ----------------------------
// 펌프 제어
// 릴레이 모듈 제어 기준
// HIGH가 켜짐인 릴레이라면 그대로 사용
// LOW가 켜짐인 릴레이라면 반대로 바꿔야 함
// ----------------------------
void runPump() {
  if (pumpRunning) return;

  if (millis() - lastPumpTime < PUMP_INTERVAL) {
    Serial.println(F("[펌프] 오늘 이미 급수함 -> 스킵"));
    waterAlert = false;
    return;
  }

  pumpRunning = true;
  lastPumpTime = millis();

  Serial.println(F("[펌프] 급수 시작"));
  digitalWrite(PUMP_PIN, HIGH);
  delay(PUMP_DURATION);
  digitalWrite(PUMP_PIN, LOW);

  pumpRunning = false;
  waterAlert = false;

  Serial.println(F("[펌프] 급수 완료"));
}

// ----------------------------
// 마스터가 상태 요청할 때 응답
// ----------------------------
void onRequest() {
  stateToSend.pumpState = pumpRunning ? 1 : 0;
  stateToSend.ledState  = ledOn ? 1 : 0;

  Wire.write((uint8_t*)&stateToSend, sizeof(stateToSend));
}

// ----------------------------
// 마스터 제어 명령 수신
// cmd = (mode << 1) | water
// ----------------------------
void onReceive(int bytes) {
  if (Wire.available()) {
    uint8_t cmd = Wire.read();
    energyMode = (cmd >> 1) & 0x03;
    waterAlert = (cmd & 0x01) ? true : false;
    printFlag = true;
  }
}

void setup() {
  Serial.begin(9600);

  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);

  strip.begin();
  strip.setBrightness(100);
  strip.clear();
  strip.show();

  memset(&stateToSend, 0, sizeof(stateToSend));

  Wire.begin(SLAVE_ADDR);
  Wire.onRequest(onRequest);
  Wire.onReceive(onReceive);

  Serial.println(F("[SLAVE] 시작!"));
}

void loop() {
  if (printFlag) {
    printFlag = false;
    Serial.print(F("[마스터 수신] mode="));
    Serial.print(energyMode);
    Serial.print(F(" water="));
    Serial.println(waterAlert ? 1 : 0);
  }

  controlLED(energyMode);

  if (waterAlert && !pumpRunning) {
    runPump();
  }

  delay(100);
}