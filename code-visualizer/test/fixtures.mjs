export const TORTURE = `
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);
int level = 0;
volatile int ticks;

void tick() { ticks++; delay(5); Serial.println("tick"); }

int classify(int v) {
  if (v < 10) return 0;
  else if (v < 20) return 1;
  else if (v < 30) { return 2; }
  return 3;
}

void setup() {
  Serial.begin(9600);
  lcd.begin(16, 2);
  Wire.begin();
  attachInterrupt(0, tick, CHANGE);
}

void loop() {
  int raw = analogRead(A0);
  level = classify(raw);
  if (level == 0) {
    lcd.print("low");
  } else if (level == 1) {
    lcd.print("mid");
    if (raw > 500) { digitalWrite(3, HIGH); } else { digitalWrite(3, LOW); }
  } else {
    lcd.print("high");
  }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 1) continue;
      if (i == 2) break;
      analogWrite(5, i * j);
    }
  }
  do {
    raw = analogRead(A1);
    if (raw > 1000) break;
    delay(10);
  } while (raw < 900);
  for (;;) {
    if (level > 2) return;
    level++;
  }
}
`;

export const SNIPPETS = {
  empty: '',
  garbage: 'void loop( { if ((( } }} else',
  globalsOnly: '#define A 1\nint x = 2;\n',
  emptyBodies: 'void setup(){} void loop(){ if (x) {} else {} while(1){} switch(a){} }',
  loopIfOnlyElse: 'void loop(){ if (a) {} else { b(); } }',
  whileBusy: 'void loop(){ while(!digitalRead(2)); digitalWrite(13, HIGH); }',
  nestedSwitch: 'void loop(){ switch(a){ case 1: switch(b){ case 2: x(); break; default: y(); } break; case 3: z(); } }',
  ifdef: 'void loop(){\n#ifdef DEBUG\n Serial.println(1);\n#else\n foo();\n#endif\n bar(); }',
  unreachable: 'int f(){ return 1; delay(5); }',
};
