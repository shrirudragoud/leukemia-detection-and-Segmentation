// Keys are stable ids (used by tests); LABELS are what kids see in the picker.
export const LABELS = {
  Blink: '💡 Blink a light',
  'Button + LED': '🔘 Button toggles a light',
  'Servo sweep': '🦾 Robot arm sweep',
  'Traffic light': '🚦 Traffic light',
  'Parking sensor': '🚗 Parking sensor (advanced)',
};

export const SAMPLES = {
  'Blink': `// Blink: turn a light on and off, again and again
#define LED_PIN 13

void setup() {
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  delay(1000);
  digitalWrite(LED_PIN, LOW);
  delay(1000);
}
`,

  'Button + LED': `const int buttonPin = 2;
const int ledPin = 13;
int lastState = LOW;
int ledState = LOW;

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  int reading = digitalRead(buttonPin);
  if (reading == LOW && lastState == HIGH) {
    ledState = !ledState;
    digitalWrite(ledPin, ledState);
    Serial.println("Toggled");
    delay(50); // debounce
  }
  lastState = reading;
}
`,

  'Servo sweep': `#include <Servo.h>

Servo myServo;
int pos = 0;

void setup() {
  myServo.attach(9);
}

void loop() {
  for (pos = 0; pos <= 180; pos += 1) {
    myServo.write(pos);
    delay(15);
  }
  for (pos = 180; pos >= 0; pos -= 1) {
    myServo.write(pos);
    delay(15);
  }
}
`,

  'Traffic light': `// A little traffic light that counts down before it changes
const int redLed = 10;
const int yellowLed = 11;
const int greenLed = 12;

void setup() {
  pinMode(redLed, OUTPUT);
  pinMode(yellowLed, OUTPUT);
  pinMode(greenLed, OUTPUT);
}

void blinkYellow(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(yellowLed, HIGH);
    delay(300);
    digitalWrite(yellowLed, LOW);
    delay(300);
  }
}

void loop() {
  digitalWrite(redLed, HIGH);
  delay(3000);
  digitalWrite(redLed, LOW);

  blinkYellow(3);

  digitalWrite(greenLed, HIGH);
  delay(3000);
  digitalWrite(greenLed, LOW);
}
`,

  'Parking sensor': `#include <Servo.h>

#define TRIG 7
#define ECHO 8
#define BUZZER 6
const int resetBtn = 2;
const int ledRed = 10;
const int ledGreen = 11;
volatile bool reset = false;
int mode = 0;
Servo gate;

void onReset() {
  reset = true;
}

long readDistanceCm() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long duration = pulseIn(ECHO, HIGH);
  return duration / 58;
}

void alarm(int times) {
  for (int i = 0; i < times; i++) {
    tone(BUZZER, 1000);
    delay(100);
    noTone(BUZZER);
    delay(100);
  }
}

void setup() {
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(ledRed, OUTPUT);
  pinMode(ledGreen, OUTPUT);
  pinMode(resetBtn, INPUT_PULLUP);
  Serial.begin(115200);
  gate.attach(9);
  attachInterrupt(digitalPinToInterrupt(resetBtn), onReset, FALLING);
}

void loop() {
  if (reset) {
    mode = 0;
    reset = false;
    Serial.println("Reset");
  }

  long d = readDistanceCm();
  Serial.println(d);

  switch (mode) {
    case 0:
      if (d < 30) {
        mode = 1;
        gate.write(90);
      }
      break;
    case 1:
    case 2:
      digitalWrite(ledRed, HIGH);
      digitalWrite(ledGreen, LOW);
      alarm(3);
      if (d > 60) {
        mode = 0;
        gate.write(0);
        break;
      }
      mode = 2;
      break;
    default:
      mode = 0;
  }

  while (d < 5) {
    d = readDistanceCm();
    if (d < 2) {
      Serial.println("Too close!");
      break;
    }
    delay(20);
  }
  delay(100);
}
`,
};
