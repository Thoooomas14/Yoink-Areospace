#include <Arduino.h>
#include <Arduino_LSM6DS3.h>

// --- PINS ---
const uint8_t SIGNAL_A_L = 2; const uint8_t SIGNAL_B_L = 3;
const uint8_t SIGNAL_A_R = 11; const uint8_t SIGNAL_B_R = 12;
static const uint8_t M_EA_R = 5; static const uint8_t M_I3 = 6; static const uint8_t M_I4 = 7;
static const uint8_t M_EA_L = 10; static const uint8_t M_I1 = 8; static const uint8_t M_I2 = 9;

// --- CONSTANTS ---
const int TPR = 3000;
const double p = 0.0625; 
const float TRACK_WIDTH = 0.2775f;
const float LOOP_HZ = 50.0f;
const float LOOP_TIME = 1.0f / LOOP_HZ;

// --- GLOBALS ---
volatile long encoder_ticks_L = 0;
volatile long encoder_ticks_R = 0;
float integral_L = 0, lastError_L = 0;
float integral_R = 0, lastError_R = 0;

float current_heading = 0.0; 
float current_position = 0.0;
double Z_offset = 0.0;

unsigned long previousMicros = 0;
unsigned long stateStartTime = 0;
unsigned long lastPrintTime = 0;
int state = 0;

// --- FILTERS ---
float smooth_L = 0, smooth_R = 0;
const float alpha = 0.15; // Lower = smoother but slower response

void decodeEncoderTicksL() {
    if (digitalRead(SIGNAL_B_L) == LOW) encoder_ticks_L--;
    else encoder_ticks_L++;
}
void decodeEncoderTicksR() {
    if (digitalRead(SIGNAL_B_R) == LOW) encoder_ticks_R++;
    else encoder_ticks_R--;
}

float PIDControl(float actual, float setpoint, float Kp, float Ki, float Kd, float &integral, float &lastErr, float dt) {
    float error = setpoint - actual;
    if (abs(setpoint) > 0.001) {
        integral += error * dt;
    } else {
        integral *= 0.8; // Quickly drain integral when stopping to prevent "creep"
    }
    integral = constrain(integral, -40, 40); 
    float derivative = (error - lastErr) / dt;
    lastErr = error;
    return (Kp * error) + (Ki * integral) + (Kd * derivative);
}

void driveMotor(uint8_t lp, uint8_t in1, uint8_t in2, float output) {
    // DEAD BAND: Crucial to stop jittering at zero
    if (abs(output) < 12) { 
        digitalWrite(in1, LOW); digitalWrite(in2, LOW);
        analogWrite(lp, 0);
        return;
    }
    int pwm = constrain((int)round(output), -255, 255);
    if (pwm > 0) {
        digitalWrite(in1, LOW); digitalWrite(in2, HIGH);
        analogWrite(lp, pwm);
    } else {
        digitalWrite(in1, HIGH); digitalWrite(in2, LOW);
        analogWrite(lp, abs(pwm));
    }
}

void setup() {
    Serial.begin(115200);
    IMU.begin();
    
    pinMode(SIGNAL_A_L, INPUT_PULLUP); pinMode(SIGNAL_B_L, INPUT_PULLUP);
    pinMode(SIGNAL_A_R, INPUT_PULLUP); pinMode(SIGNAL_B_R, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(SIGNAL_A_L), decodeEncoderTicksL, RISING);
    attachInterrupt(digitalPinToInterrupt(SIGNAL_A_R), decodeEncoderTicksR, RISING);

    pinMode(M_EA_L, OUTPUT); pinMode(M_I1, OUTPUT); pinMode(M_I2, OUTPUT);
    pinMode(M_EA_R, OUTPUT); pinMode(M_I3, OUTPUT); pinMode(M_I4, OUTPUT);

    // Simple Gyro Calibration
    float ox, oy, oz;
    for (int i = 0; i < 200; i++) {
        if(IMU.gyroscopeAvailable()) IMU.readGyroscope(ox, oy, oz);
        Z_offset -= oz;
        delay(5);
    }
    Z_offset /= 200.0;

    previousMicros = micros();
    stateStartTime = millis();
}

void loop() {
    unsigned long currentMicros = micros();
    float dt = (currentMicros - previousMicros) / 1000000.0f;
    
    // Maintain a steady loop frequency (50Hz)
    if (dt < LOOP_TIME) return; 
    previousMicros = currentMicros;

    // 1. Process Encoders
    noInterrupts();
    long tL = encoder_ticks_L; long tR = encoder_ticks_R;
    encoder_ticks_L = 0; encoder_ticks_R = 0;
    interrupts();

    float raw_L = ((2.0 * PI * tL / TPR) / dt) * p;
    float raw_R = ((2.0 * PI * tR / TPR) / dt) * p;

    // Smoothing speed to stop jitter
    smooth_L = (alpha * raw_L) + ((1.0 - alpha) * smooth_L);
    smooth_R = (alpha * raw_R) + ((1.0 - alpha) * smooth_R);

    // 2. Update Position and Heading (Odometry)
    float gx, gy, gz = 0;
    if (IMU.gyroscopeAvailable()) IMU.readGyroscope(gx, gy, gz);
    float corrected_gz = gz + Z_offset;

    // Integrate for Heading (Degrees) and Position (Meters)
    current_heading += corrected_gz * dt;
    current_position += ((raw_L + raw_R) / 2.0f) * dt; // Use raw for distance to avoid filter lag

    // 3. State Machine Logic
    float targetV = 0, targetYawDeg = 0;

    switch (state) {
        case 0: // FORWARD 1 METER
            targetV = 0.5f;
            targetYawDeg = 0.0f;
            if (current_position >= 1.0f) {
                state = 1;
                stateStartTime = millis();
                current_position = 0; // Reset for next leg
            }
            break;

        case 1: // DELAY 2 SECONDS
            targetV = 0.0f;
            targetYawDeg = 0.0f;
            if (millis() - stateStartTime >= 2000) {
                state = 2;
                current_heading = 0; // Reset heading so we can turn 160 from 'now'
            }
            break;

        case 2: // ROTATE 160 DEGREES
            targetV = 0.0f;
            targetYawDeg = 90.0f; // Turn speed in deg/s
            if (abs(current_heading) >= 160.0f) {
                state = 0;
                current_position = 0;
            }
            break;
    }

    // 4. Differential Drive & PID
    float targetYawRad = (targetYawDeg * PI) / 180.0f;
    float tL_speed = targetV - (targetYawRad * TRACK_WIDTH / 2.0f);
    float tR_speed = targetV + (targetYawRad * TRACK_WIDTH / 2.0f);

    // Adjusted Gains (Kp=3.5, Ki=0.15) for stability
    float outL = PIDControl(smooth_L, tL_speed, 5.0, 0.15, 0.01, integral_L, lastError_L, dt);
    float outR = PIDControl(smooth_R, tR_speed, 5.0, 0.15, 0.01, integral_R, lastError_R, dt);

    driveMotor(M_EA_L, M_I1, M_I2, outL * 255.0f);
    driveMotor(M_EA_R, M_I3, M_I4, outR * 255.0f);

    // 5. Diagnostics
    if (millis() - lastPrintTime >= 250) {
        Serial.print("State: "); Serial.print(state);
        Serial.print(" | Pos: "); Serial.print(current_position);
        Serial.print(" | Head: "); Serial.print(current_heading);
        Serial.print(" | L_Spd: "); Serial.println(smooth_L);
        lastPrintTime = millis();
    }
}