#include <Wire.h>
#include <DHT.h>
#include <Adafruit_INA219.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>

#define DHT_PIN           5
#define DHT_TYPE          DHT22   // DHT11이면 DHT11로 변경

#define NEOPIXEL_PIN      6
#define NUM_PIXELS        60

#define PUMP_PIN          8       // MOSFET Gate

#define LDR_PIN           A1
#define SOIL_PIN          A2

#define SEND_INTERVAL     10000UL

#define BAT_MAX_V         16.8
#define BAT_MIN_V         12.0

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_INA219 ina219_solar(0x40);
Adafruit_INA219 ina219_battery(0x41);
Adafruit_NeoPixel strip(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

unsigned long lastSendTime = 0;
unsigned long lastPrintTime = 0;

bool solarINAOK = false;
bool batteryINAOK = false;

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

// 조도가 높을수록 네오픽셀은 어둡게
int calcLedBrightness(int light, uint8_t mode) {
  if (mode == 0) return 0;

  // light 0~100 -> brightness 255~30
  int brightness = map(light, 0, 100, 255, 30);

  if (mode == 1) {
    brightness = brightness / 2;   // 절약모드면 절반
  }

  brightness = constrain(brightness, 0, 255);
  return brightness;
}

void updateNeoPixelPurple(int brightness) {
  strip.setBrightness(brightness);

  for (int i = 0; i < NUM_PIXELS; i++) {
    // 보라색
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

  solarINAOK = ina219_solar.begin();
  if (solarINAOK) {
    Serial.println(F("[INA219] solar OK"));
  } else {
    Serial.println(F("[INA219] solar FAIL"));
  }

  batteryINAOK = ina219_battery.begin();
  if (batteryINAOK) {
    Serial.println(F("[INA219] battery OK"));
  } else {
    Serial.println(F("[INA219] battery FAIL"));
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
  // INA219 - 태양광
  // ----------------------------
  float solar_busV = 0.0;
  float solar_shuntV = 0.0;
  float solar_current_mA = 0.0;
  float solar_power_mW = 0.0;

  float solar_voltage_V = 0.0;
  float solar_current_A = 0.0;
  float solar_power_W = 0.0;

  if (solarINAOK) {
    solar_busV = sanitizeFloat(ina219_solar.getBusVoltage_V());
    solar_shuntV = sanitizeFloat(ina219_solar.getShuntVoltage_mV());
    solar_current_mA = sanitizeFloat(ina219_solar.getCurrent_mA());
    solar_power_mW = sanitizeFloat(ina219_solar.getPower_mW());

    solar_voltage_V = sanitizeFloat(solar_busV + (solar_shuntV / 1000.0));
    solar_current_A = sanitizeFloat(solar_current_mA / 1000.0);
    solar_power_W = sanitizeFloat(solar_power_mW / 1000.0);
  }

  // ----------------------------
  // INA219 - 배터리
  // ----------------------------
  float battery_busV = 0.0;
  float battery_shuntV = 0.0;
  float battery_current_mA = 0.0;
  float battery_power_mW = 0.0;

  float battery_voltage_V = 0.0;
  float battery_current_A = 0.0;
  float battery_power_W = 0.0;

  if (batteryINAOK) {
    battery_busV = sanitizeFloat(ina219_battery.getBusVoltage_V());
    battery_shuntV = sanitizeFloat(ina219_battery.getShuntVoltage_mV());
    battery_current_mA = sanitizeFloat(ina219_battery.getCurrent_mA());
    battery_power_mW = sanitizeFloat(ina219_battery.getPower_mW());

    battery_voltage_V = sanitizeFloat(battery_busV + (battery_shuntV / 1000.0));
    battery_current_A = sanitizeFloat(battery_current_mA / 1000.0);
    battery_power_W = sanitizeFloat(battery_power_mW / 1000.0);
  }

  // ----------------------------
  // 상태 판단
  // ----------------------------
  int soc = estimateSOC(battery_voltage_V);
  uint8_t mode = calcMode(soc);
  bool waterAlert = calcWaterAlert(soil);

  // ----------------------------
  // 제어 로직
  // ----------------------------
  // 펌프: 토양습도 부족 + SOC 20 초과일 때 ON
  bool pumpOn = (waterAlert && soc > 20);

  // 네오픽셀 밝기: 조도 + mode 기반
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