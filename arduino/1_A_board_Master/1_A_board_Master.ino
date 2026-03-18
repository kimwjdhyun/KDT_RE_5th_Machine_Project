#include <Wire.h>
#include <DHT.h>
#include <Adafruit_INA3221.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>

#define DHT_PIN           5
#define DHT_TYPE          DHT22

#define NEOPIXEL_PIN      6
#define NUM_PIXELS        60

#define PUMP_PIN          8

#define LDR_PIN           A1
#define SOIL_PIN          A2

#define SEND_INTERVAL     10000UL

#define BAT_MAX_V         16.8
#define BAT_MIN_V         10.8

// INA3221 채널 번호
#define INA_CH_SOLAR      0
#define INA_CH_BATTERY    1
#define INA_CH_SPARE      2

// LED 히스테리시스 기준
#define LED_ON_THRESHOLD   65
#define LED_OFF_THRESHOLD  75

// 토양센서 보정값
#define SOIL_WET           50   // 아주 젖은 상태
#define SOIL_DRY          150   // 마른 상태

// 급수 설정
#define WATER_THRESHOLD    40   // soil %가 40 미만이면 급수 후보
#define PUMP_DURATION_MS 3000UL // 3초 급수
#define PUMP_COOLDOWN_MS 60000UL // 급수 후 60초 대기

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_INA3221 ina3221;
Adafruit_NeoPixel strip(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

unsigned long lastSendTime = 0;
unsigned long lastPrintTime = 0;

bool ina3221OK = false;
bool ledOnState = false;

// 펌프 상태 관리
bool pumpRunning = false;
unsigned long pumpStartTime = 0;
unsigned long lastPumpEndTime = 0;

// ----------------------------
// 유틸
// ----------------------------
float sanitizeFloat(float value) {
  if (isnan(value) || isinf(value)) return 0.0;
  return value;
}

int calcLight() {
  return map(analogRead(LDR_PIN), 0, 1023, 0, 100);
}

// ----------------------------
// 토양센서
// ----------------------------
int readSoilRaw() {
  return analogRead(SOIL_PIN);
}

int calcSoilPercent(int raw) {
  raw = constrain(raw, SOIL_WET, SOIL_DRY);
  return map(raw, SOIL_DRY, SOIL_WET, 0, 100);
}

int estimateSOC(float voltage) {
  float soc = (voltage - BAT_MIN_V) / (BAT_MAX_V - BAT_MIN_V) * 100.0;
  if (soc < 0) soc = 0;
  if (soc > 100) soc = 100;
  return (int)soc;
}

// 0: 긴급절전, 1: 절약모드, 2: 풀가동
uint8_t calcMode(int soc) {
  if (soc < 20) return 0;
  if (soc < 60) return 1;
  return 2;
}

bool calcWaterAlert(int soilPercent) {
  return soilPercent < WATER_THRESHOLD;
}

// ----------------------------
// LED 히스테리시스
// ----------------------------
bool shouldLedOnHysteresis(int light, uint8_t mode) {
  if (mode == 0) {
    ledOnState = false;
    return ledOnState;
  }

  if (light <= LED_ON_THRESHOLD) {
    ledOnState = true;
  } else if (light >= LED_OFF_THRESHOLD) {
    ledOnState = false;
  }
  return ledOnState;
}

void updateNeoPixelPurple(bool ledOn) {
  int brightness = ledOn ? 100 : 0;

  strip.setBrightness(brightness);

  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, ledOn ? strip.Color(180, 0, 255) : 0);
  }

  strip.show();
}

// ----------------------------
// 펌프 제어
// ----------------------------
void updatePumpControl(bool waterAlert, int soc, float battery_voltage_V) {
  unsigned long now = millis();

  // 이미 펌프 작동 중이면 3초 후 종료
  if (pumpRunning) {
    if (now - pumpStartTime >= PUMP_DURATION_MS) {
      digitalWrite(PUMP_PIN, LOW);
      pumpRunning = false;
      lastPumpEndTime = now;
    }
    return;
  }

  // 배터리 측정 실패면 급수 금지
  if (battery_voltage_V < 1.0) {
    digitalWrite(PUMP_PIN, LOW);
    return;
  }

  // SOC 낮으면 급수 금지
  if (soc <= 20) {
    digitalWrite(PUMP_PIN, LOW);
    return;
  }

  // 쿨다운 중이면 급수 금지
  if (now - lastPumpEndTime < PUMP_COOLDOWN_MS) {
    digitalWrite(PUMP_PIN, LOW);
    return;
  }

  // 토양 부족하면 3초 급수 시작
  if (waterAlert) {
    digitalWrite(PUMP_PIN, HIGH);
    pumpRunning = true;
    pumpStartTime = now;
  } else {
    digitalWrite(PUMP_PIN, LOW);
  }
}

// ----------------------------
// setup
// ----------------------------
void setup() {
  Serial.begin(9600);
  dht.begin();
  Wire.begin();

  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);

  pinMode(SOIL_PIN, INPUT);
  pinMode(LDR_PIN, INPUT);

  strip.begin();
  strip.clear();
  strip.show();

  Serial.println(F("[MASTER] 시작"));

  if (!ina3221.begin(0x40)) {
    Serial.println(F("[INA3221] FAIL"));
    ina3221OK = false;
  } else {
    Serial.println(F("[INA3221] OK"));
    ina3221OK = true;
  }

  lastSendTime = millis() - SEND_INTERVAL;
  lastPrintTime = 0;
}

// ----------------------------
// loop
// ----------------------------
void loop() {
  // ----------------------------
  // 센서 읽기
  // ----------------------------
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temperature)) temperature = 0.0;
  if (isnan(humidity)) humidity = 0.0;

  int light = calcLight();

  int soilRaw = readSoilRaw();
  int soil = calcSoilPercent(soilRaw);

  // ----------------------------
  // INA3221 - 태양광 / 배터리
  // ----------------------------
  float solar_voltage_V = 0.0;
  float solar_current_A = 0.0;
  float solar_power_W   = 0.0;

  float battery_voltage_V = 0.0;
  float battery_current_A = 0.0;
  float battery_power_W   = 0.0;

  if (ina3221OK) {
    solar_voltage_V = sanitizeFloat(ina3221.getBusVoltage(INA_CH_SOLAR));
    solar_current_A = sanitizeFloat(ina3221.getCurrentAmps(INA_CH_SOLAR));
    solar_power_W   = sanitizeFloat(solar_voltage_V * solar_current_A);

    battery_voltage_V = sanitizeFloat(ina3221.getBusVoltage(INA_CH_BATTERY));
    battery_current_A = sanitizeFloat(ina3221.getCurrentAmps(INA_CH_BATTERY));
    battery_power_W   = sanitizeFloat(battery_voltage_V * battery_current_A);
  }

  // ----------------------------
  // 상태 판단
  // ----------------------------
  int soc = estimateSOC(battery_voltage_V);

  uint8_t mode;
  if (battery_voltage_V < 1.0) {
    mode = 0;  // 측정 실패 시 보수적으로 절전
  } else {
    mode = calcMode(soc);
  }

  bool waterAlert = calcWaterAlert(soil);

  // ----------------------------
  // 제어 로직
  // ----------------------------
  updatePumpControl(waterAlert, soc, battery_voltage_V);

  bool ledOn = shouldLedOnHysteresis(light, mode);
  int ledBrightness = ledOn ? 100 : 0;

  updateNeoPixelPurple(ledOn);

  int pumpState = digitalRead(PUMP_PIN);
  int ledState = ledOn ? 1 : 0;

  // ----------------------------
  // 디버그 출력
  // ----------------------------
  if (millis() - lastPrintTime >= 10000UL) {
    lastPrintTime = millis();

    Serial.print(F("[DEBUG] T="));
    Serial.print(temperature, 1);
    Serial.print(F(" H="));
    Serial.print(humidity, 1);
    Serial.print(F(" L="));
    Serial.print(light);

    Serial.print(F(" SOIL_RAW="));
    Serial.print(soilRaw);
    Serial.print(F(" SOIL="));
    Serial.print(soil);

    Serial.print(F(" SV="));
    Serial.print(solar_voltage_V, 2);
    Serial.print(F(" SI="));
    Serial.print(solar_current_A, 3);
    Serial.print(F(" SP="));
    Serial.print(solar_power_W, 3);

    Serial.print(F(" BV="));
    Serial.print(battery_voltage_V, 2);
    Serial.print(F(" BI="));
    Serial.print(battery_current_A, 3);
    Serial.print(F(" BP="));
    Serial.print(battery_power_W, 3);

    Serial.print(F(" SOC="));
    Serial.print(soc);
    Serial.print(F(" MODE="));
    Serial.print(mode);
    Serial.print(F(" WATER="));
    Serial.print(waterAlert ? 1 : 0);
    Serial.print(F(" PUMP="));
    Serial.print(pumpState);
    Serial.print(F(" LED="));
    Serial.print(ledState);
    Serial.print(F(" LED_B="));
    Serial.println(ledBrightness);
  }

  // ----------------------------
  // Flask용 JSON 출력
  // ----------------------------
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();

    Serial.print("{");
    Serial.print("\"temperature\":"); Serial.print(temperature, 1); Serial.print(",");
    Serial.print("\"humidity\":"); Serial.print(humidity, 1); Serial.print(",");
    Serial.print("\"soil_raw\":"); Serial.print(soilRaw); Serial.print(",");
    Serial.print("\"soil\":"); Serial.print(soil); Serial.print(",");
    Serial.print("\"light\":"); Serial.print(light); Serial.print(",");

    Serial.print("\"solar_voltage\":"); Serial.print(solar_voltage_V, 2); Serial.print(",");
    Serial.print("\"solar_current\":"); Serial.print(solar_current_A, 3); Serial.print(",");
    Serial.print("\"solar_power\":"); Serial.print(solar_power_W, 3); Serial.print(",");

    Serial.print("\"battery_voltage\":"); Serial.print(battery_voltage_V, 2); Serial.print(",");
    Serial.print("\"battery_current\":"); Serial.print(battery_current_A, 3); Serial.print(",");
    Serial.print("\"battery_power\":"); Serial.print(battery_power_W, 3); Serial.print(",");

    Serial.print("\"pump\":"); Serial.print(pumpState); Serial.print(",");
    Serial.print("\"led\":"); Serial.print(ledState); Serial.print(",");
    Serial.print("\"led_brightness\":"); Serial.print(ledBrightness); Serial.print(",");
    Serial.print("\"soc\":"); Serial.print(soc); Serial.print(",");
    Serial.print("\"mode\":"); Serial.print(mode); Serial.print(",");
    Serial.print("\"water_alert\":"); Serial.print(waterAlert ? 1 : 0);
    Serial.println("}");
  }

  delay(100);
}