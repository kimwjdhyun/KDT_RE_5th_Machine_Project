#include <Wire.h>
#include <Adafruit_INA219.h>
#include <Adafruit_NeoPixel.h>

#define LED_PIN       7
#define PUMP_PIN      9
#define NUMPIXELS     60
#define SLAVE_ADDR    8
#define PUMP_DURATION 1000
#define PUMP_INTERVAL 86400000UL

#define BAT_MAX_V  16.8
#define BAT_MIN_V  12.0

struct PowerData {
  int voltage_x100;      // V * 100
  int current_mA_x10;    // mA * 10
  int power_mW_x10;      // mW * 10
  int pumpState;
  int ledState;
  int soc;
};

Adafruit_INA219   ina219_solar(0x40);
Adafruit_INA219   ina219_battery(0x41);
Adafruit_NeoPixel strip(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

PowerData     dataToSend;
int           energyMode   = 2;
bool          waterAlert   = false;
bool          pumpRunning  = false;
bool          ledOn        = false;
unsigned long lastPumpTime = 0;


int estimateSOC(float voltage) {
  float soc = (voltage - BAT_MIN_V) / (BAT_MAX_V - BAT_MIN_V) * 100.0;
  return (int)constrain(soc, 0, 100);
}


void controlLED(int mode) {
  int brightness = 0;

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


void runPump() {
  if (pumpRunning) return;

  if (millis() - lastPumpTime < PUMP_INTERVAL) {
    Serial.println("[펌프] 오늘 이미 급수함 -> 스킵");
    waterAlert = false;
    return;
  }

  pumpRunning  = true;
  lastPumpTime = millis();

  Serial.println("[펌프] 급수 시작");
  digitalWrite(PUMP_PIN, HIGH);
  delay(PUMP_DURATION);
  digitalWrite(PUMP_PIN, LOW);

  pumpRunning = false;
  waterAlert  = false;

  Serial.println("[펌프] 급수 완료");
}


void onRequest() {
  Wire.write((uint8_t*)&dataToSend, sizeof(dataToSend));
}


void onReceive(int bytes) {
  if (Wire.available()) {
    byte cmd   = Wire.read();
    energyMode = (cmd >> 1) & 0x03;
    waterAlert = cmd & 0x01;

    Serial.print("[A수신] mode: ");
    Serial.print(energyMode);
    Serial.print(" / water: ");
    Serial.println(waterAlert ? "true" : "false");
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

  if (!ina219_solar.begin()) {
    Serial.println("태양광 INA219(0x40) 연결 실패!");
  } else {
    Serial.println("태양광 INA219(0x40) 연결 성공!");
  }

  if (!ina219_battery.begin()) {
    Serial.println("배터리 INA219(0x41) 연결 실패!");
  } else {
    Serial.println("배터리 INA219(0x41) 연결 성공!");
  }

  Wire.begin(SLAVE_ADDR);
  Wire.onRequest(onRequest);
  Wire.onReceive(onReceive);

  memset(&dataToSend, 0, sizeof(dataToSend));

  Serial.println("[SLAVE] 시작!");
}


void loop() {
  // 태양광 INA219 측정
  float busVoltage   = ina219_solar.getBusVoltage_V();
  float shuntVoltage = ina219_solar.getShuntVoltage_mV();
  float current_mA   = ina219_solar.getCurrent_mA();
  float power_mW     = ina219_solar.getPower_mW();
  float loadVoltage  = busVoltage + (shuntVoltage / 1000.0);

  Serial.print("[태양광] 전압: ");
  Serial.print(loadVoltage, 2);
  Serial.print(" V / 전류: ");
  Serial.print(current_mA, 1);
  Serial.print(" mA / 전력: ");
  Serial.print(power_mW, 1);
  Serial.println(" mW");

  // 배터리 INA219 측정
  float batBusV    = ina219_battery.getBusVoltage_V();
  float batShuntV  = ina219_battery.getShuntVoltage_mV();
  float batVoltage = batBusV + (batShuntV / 1000.0);
  int   soc        = estimateSOC(batVoltage);

  Serial.print("[배터리] 전압: ");
  Serial.print(batVoltage, 2);
  Serial.print(" V / SOC: ");
  Serial.print(soc);
  Serial.println("%");

  // 서버에서 내려준 모드 기준 LED 제어
  controlLED(energyMode);

  // 서버에서 water_alert가 내려왔을 때만 급수
  if (waterAlert && !pumpRunning) {
    runPump();
  }

  // 마스터로 전달할 데이터 준비
  dataToSend.voltage_x100   = (int)(loadVoltage * 100);
  dataToSend.current_mA_x10 = (int)(current_mA * 10);
  dataToSend.power_mW_x10   = (int)(power_mW * 10);
  dataToSend.pumpState      = pumpRunning ? 1 : 0;
  dataToSend.ledState       = ledOn ? 1 : 0;
  dataToSend.soc            = soc;

  delay(1000);
}
