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

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_INA3221 ina3221;
Adafruit_NeoPixel strip(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

unsigned long lastSendTime = 0;
unsigned long lastPrintTime = 0;

bool ina3221OK = false;

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

int calcSoil() {
  int raw = constrain(analogRead(SOIL_PIN), 300, 1023);
  return map(raw, 300, 1023, 100, 0);
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

bool calcWaterAlert(int soil) {
  return soil < 40;
}

// 스마트팜용 LED 로직
// 밝으면 꺼지고, 어두우면 켜짐
int calcLedBrightness(int light, uint8_t mode) {
  // 배터리 매우 부족하면 최소 밝기만 유지
  if (mode == 0) {
    if (light <= 70) return 30;
    return 0;
  }

  // 밝으면 LED OFF
  if (light > 70) return 0;

  // 어두울수록 밝게
  int brightness = map(light, 0, 70, 255, 50);

  if (mode == 1) {
    brightness /= 2;
  }

  return constrain(brightness, 0, 255);
}

void updateNeoPixelPurple(int brightness) {
  strip.setBrightness(brightness);

  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, strip.Color(180, 0, 255));
  }

  strip.show();
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

  strip.begin();
  strip.clear();
  strip.show();

  Serial.println(F("[MASTER] 시작"));

  // INA3221 시작
  // 기본 주소는 보통 0x40
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
  int soil = calcSoil();

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
    // 채널 1: 태양광
    solar_voltage_V = sanitizeFloat(ina3221.getBusVoltage(INA_CH_SOLAR));
    solar_current_A = sanitizeFloat(ina3221.getCurrentAmps(INA_CH_SOLAR));
    solar_power_W   = sanitizeFloat(solar_voltage_V * solar_current_A);

    // 채널 2: 배터리
    battery_voltage_V = sanitizeFloat(ina3221.getBusVoltage(INA_CH_BATTERY));
    battery_current_A = sanitizeFloat(ina3221.getCurrentAmps(INA_CH_BATTERY));
    battery_power_W   = sanitizeFloat(battery_voltage_V * battery_current_A);
  }

  // ----------------------------
  // 상태 판단
  // ----------------------------
  int soc = estimateSOC(battery_voltage_V);

  // 배터리 측정이 아직 0 근처면 테스트용으로 풀모드
  uint8_t mode;
  if (battery_voltage_V < 1.0) {
    mode = 2;
  } else {
    mode = calcMode(soc);
  }

  bool waterAlert = calcWaterAlert(soil);

  // ----------------------------
  // 제어 로직
  // ----------------------------
  // 펌프: 토양습도 부족 + SOC 20 초과일 때 ON
  bool pumpOn = (waterAlert && soc > 20);

  // 배터리 측정이 없으면 펌프는 안전상 OFF
  if (battery_voltage_V < 1.0) {
    pumpOn = false;
  }

  int ledBrightness = calcLedBrightness(light, mode);

  digitalWrite(PUMP_PIN, pumpOn ? HIGH : LOW);
  updateNeoPixelPurple(ledBrightness);

  int pumpState = digitalRead(PUMP_PIN);
  int ledState = (ledBrightness > 0) ? 1 : 0;

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
    Serial.print(F(" S="));
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