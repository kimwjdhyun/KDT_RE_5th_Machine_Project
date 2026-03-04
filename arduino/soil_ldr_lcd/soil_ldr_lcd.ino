// soil_ldr_lcd.ino
// 토양습도 + LDR 조도 센서 + LCD
// 저항: LDR → 10kΩ 풀다운 필수

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ─────────────────────────────────────────
// 핀 설정
// ─────────────────────────────────────────
#define SOIL_PIN  A2   // 토양습도
#define LDR_PIN   A1   // 조도

LiquidCrystal_I2C lcd(0x27, 16, 2);


// ─────────────────────────────────────────
// 토양습도 계산 (0~100%)
// 300(촉촉) ~ 1023(건조) → 100~0% 반전
// ─────────────────────────────────────────
int calcSoil() {
  int raw     = analogRead(SOIL_PIN);
  int clamped = constrain(raw, 300, 1023);
  return map(clamped, 300, 1023, 100, 0);
}


// ─────────────────────────────────────────
// 조도 계산 (0~100%)
// ─────────────────────────────────────────
int calcLight() {
  int raw = analogRead(LDR_PIN);
  return map(raw, 0, 1023, 0, 100);
}


void setup() {
  Serial.begin(9600);

  Wire.begin();
  lcd.init();
  lcd.backlight();

  lcd.setCursor(0, 0);
  lcd.print("SmartFarm Ready!");
  lcd.setCursor(0, 1);
  lcd.print("Loading...");
  delay(1500);
  lcd.clear();

  Serial.println("토양습도 + 조도 + LCD 시작!");
}


void loop() {
  int soil  = calcSoil();
  int light = calcLight();

  // ── 시리얼 출력 ──
  Serial.print("토양습도: "); Serial.print(soil);
  Serial.print("%  조도: "); Serial.print(light);
  Serial.println("%");

  // ── LCD 줄 1: 조도 ──
  lcd.setCursor(0, 0);
  lcd.print("Light:");
  lcd.print(light);
  lcd.print("%  ");
  // 낮/밤 표시
  if (light >= 30) {
    lcd.print("DAY ");
  } else {
    lcd.print("NIGHT");
  }

  // ── LCD 줄 2: 토양습도 + 상태 ──
  lcd.setCursor(0, 1);
  lcd.print("Soil:");
  lcd.print(soil);
  lcd.print("% ");
  if (soil >= 60) {
    lcd.print("WET  ");
  } else if (soil >= 40) {
    lcd.print("OK   ");
  } else {
    lcd.print("DRY! ");
  }

  delay(1000);
}
