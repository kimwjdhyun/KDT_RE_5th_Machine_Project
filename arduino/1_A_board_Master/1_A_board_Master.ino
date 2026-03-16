#include <Wire.h>
#include <DHT.h>
#include <Adafruit_INA219.h>
#include <math.h>
#include <string.h>

#define DHT_PIN         5
#define DHT_TYPE        DHT11
#define LDR_PIN         A1
#define SOIL_PIN        A2

#define SLAVE_ADDR      8
#define SEND_INTERVAL   10000UL

#define BAT_MAX_V       16.8
#define BAT_MIN_V       12.0

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_INA219 ina219_solar(0x40);
Adafruit_INA219 ina219_battery(0x41);

unsigned long lastSendTime = 0;
unsigned long lastPrintTime = 0;

struct DeviceState {
  uint8_t pumpState;
  uint8_t ledState;
};

DeviceState slaveState = {0, 0};

// ----------------------------
// 유틸
// ----------------------------
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

float sanitizeFloat(float value) {
  if (isnan(value) || isinf(value)) return 0.0;
  return value;
}

// 0: 절전, 1: 중간, 2: 정상
uint8_t calcMode(int soc) {
  if (soc < 20) return 0;
  if (soc < 60) return 1;
  return 2;
}

bool calcWaterAlert(int soil) {
  return soil < 40;
}

// cmd = (mode << 1) | water
void sendCommandToSlave(uint8_t mode, bool waterAlert) {
  uint8_t cmd = (mode << 1) | (waterAlert ? 1 : 0);

  Wire.beginTransmission(SLAVE_ADDR);
  Wire.write(cmd);
  Wire.endTransmission();
}

void requestSlaveState() {
  Wire.requestFrom(SLAVE_ADDR, sizeof(DeviceState));

  if (Wire.available() >= (int)sizeof(DeviceState)) {
    uint8_t *ptr = (uint8_t*)&slaveState;
    for (size_t i = 0; i < sizeof(DeviceState); i++) {
      ptr[i] = Wire.read();
    }
  }
}

void setup() {
  Serial.begin(9600);
  dht.begin();
  Wire.begin();

  Serial.println(F("[MASTER] 시작"));

  if (ina219_solar.begin()) {
    Serial.println(F("[INA219] solar OK"));
  } else {
    Serial.println(F("[INA219] solar FAIL"));
  }

  if (ina219_battery.begin()) {
    Serial.println(F("[INA219] battery OK"));
  } else {
    Serial.println(F("[INA219] battery FAIL"));
  }

  lastSendTime = millis() - SEND_INTERVAL;
  lastPrintTime = 0;
}

void loop() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temperature)) temperature = 0.0;
  if (isnan(humidity)) humidity = 0.0;

  int light = calcLight();
  int soil = calcSoil();

  // 태양광 INA219
  float solar_busV = sanitizeFloat(ina219_solar.getBusVoltage_V());
  float solar_shuntV = sanitizeFloat(ina219_solar.getShuntVoltage_mV());
  float solar_current_mA = sanitizeFloat(ina219_solar.getCurrent_mA());
  float solar_power_mW = sanitizeFloat(ina219_solar.getPower_mW());

  float solar_voltage_V = sanitizeFloat(solar_busV + (solar_shuntV / 1000.0));
  float solar_current_A = sanitizeFloat(solar_current_mA / 1000.0);
  float solar_power_W = sanitizeFloat(solar_power_mW / 1000.0);

  // 배터리 INA219
  float battery_busV = sanitizeFloat(ina219_battery.getBusVoltage_V());
  float battery_shuntV = sanitizeFloat(ina219_battery.getShuntVoltage_mV());
  float battery_current_mA = sanitizeFloat(ina219_battery.getCurrent_mA());
  float battery_power_mW = sanitizeFloat(ina219_battery.getPower_mW());

  float battery_voltage_V = sanitizeFloat(battery_busV + (battery_shuntV / 1000.0));
  float battery_current_A = sanitizeFloat(battery_current_mA / 1000.0);
  float battery_power_W = sanitizeFloat(battery_power_mW / 1000.0);

  int soc = estimateSOC(battery_voltage_V);

  uint8_t mode = calcMode(soc);
  bool waterAlert = calcWaterAlert(soil);

  // 슬레이브 제어
  sendCommandToSlave(mode, waterAlert);
  delay(20);
  requestSlaveState();

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
    Serial.print(slaveState.pumpState);
    Serial.print(F(" LED="));
    Serial.println(slaveState.ledState);
  }

  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();

    // JSON 한 줄 출력
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
    Serial.print("\"pump\":"); Serial.print(slaveState.pumpState); Serial.print(",");
    Serial.print("\"led\":"); Serial.print(slaveState.ledState); Serial.print(",");
    Serial.print("\"soc\":"); Serial.print(soc); Serial.print(",");
    Serial.print("\"mode\":"); Serial.print(mode); Serial.print(",");
    Serial.print("\"water_alert\":"); Serial.print(waterAlert ? 1 : 0);
    Serial.println("}");
  }

  delay(100);
}