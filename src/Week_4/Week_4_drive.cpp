#include <Arduino.h>
#include <Arduino_LSM6DS3.h>

// -------------------------------------------------------------- //
// CONFIGURATION
// -------------------------------------------------------------- //

// Wheel parameters
const int TPR = 3000;      // Ticks per revolution
const double RADIUS = 0.0625; // Wheel radius in meters (from your variable 'p')
// Calculate Ticks Per Meter: (3000) / (2 * pi * r)
const double TICKS_PER_METER = TPR / (2.0 * PI * RADIUS); 

// Pin Definitions
const uint8_t SIGNAL_A_L = 2;
const uint8_t SIGNAL_B_L = 3;
const uint8_t SIGNAL_A_R = 11;
const uint8_t SIGNAL_B_R = 12;

// Motor Pins
const uint8_t M_EA_L = 10; 
const uint8_t M_I1 = 8;
const uint8_t M_I2 = 9;

const uint8_t M_EA_R = 5; 
const uint8_t M_I3 = 6;
const uint8_t M_I4 = 7;

// -------------------------------------------------------------- //
// STATE MACHINE
// -------------------------------------------------------------- //
enum RobotState {
  STATE_IDLE,
  STATE_FORWARD_1,
  STATE_TURN_180,
  STATE_FORWARD_2,
  STATE_STOP
};

RobotState currentState = STATE_IDLE;

// -------------------------------------------------------------- //
// GLOBAL VARIABLES
// -------------------------------------------------------------- //
volatile long encoder_ticks_L = 0;
volatile long encoder_ticks_R = 0;

double gyro_z_offset = 0.0;
double current_heading = 0.0; // Current robot angle in degrees
unsigned long last_loop_time = 0;

// PID Variables for Heading Hold
float kp_heading = 2.5; // Proportional gain for steering
float base_speed_pwm = 100; // Base motor speed (0-255)

// -------------------------------------------------------------- //
// INTERRUPT SERVICE ROUTINES
// -------------------------------------------------------------- //
void decodeEncoderTicksL() {
    if (digitalRead(SIGNAL_B_L) == LOW) {
        encoder_ticks_L--;
    } else {
        encoder_ticks_L++;
    }
}

void decodeEncoderTicksR() {
    if (digitalRead(SIGNAL_B_R) == LOW) {
        encoder_ticks_R++;
    } else {
        encoder_ticks_R--;
    }
}

// -------------------------------------------------------------- //
// HELPER FUNCTIONS
// -------------------------------------------------------------- //

void setMotors(int leftPWM, int rightPWM) {
    // Constrain Speed
    leftPWM = constrain(leftPWM, -255, 255);
    rightPWM = constrain(rightPWM, -255, 255);

    // LEFT MOTOR
    if (leftPWM >= 0) {
        digitalWrite(M_I1, LOW);
        digitalWrite(M_I2, HIGH);
        analogWrite(M_EA_L, leftPWM);
    } else {
        digitalWrite(M_I1, HIGH);
        digitalWrite(M_I2, LOW);
        analogWrite(M_EA_L, -leftPWM);
    }

    // RIGHT MOTOR
    if (rightPWM >= 0) {
        digitalWrite(M_I3, LOW);
        digitalWrite(M_I4, HIGH);
        analogWrite(M_EA_R, rightPWM);
    } else {
        digitalWrite(M_I3, HIGH);
        digitalWrite(M_I4, LOW);
        analogWrite(M_EA_R, -rightPWM);
    }
}

void stopMotors() {
    setMotors(0, 0);
}

void calibrateGyro() {
    Serial.println("Calibrating Gyro...");
    float sumZ = 0;
    int samples = 1000;
    for (int i = 0; i < samples; i++) {
        float x, y, z;
        if (IMU.readGyroscope(x, y, z)) {
            sumZ += z;
        }
        delay(1);
    }
    gyro_z_offset = sumZ / samples;
    Serial.print("Offset Z: "); Serial.println(gyro_z_offset);
}

void resetEncoders() {
    noInterrupts();
    encoder_ticks_L = 0;
    encoder_ticks_R = 0;
    interrupts();
}

long getAvgEncoderDistance() {
    noInterrupts();
    long l = abs(encoder_ticks_L);
    long r = abs(encoder_ticks_R);
    interrupts();
    return (l + r) / 2;
}

// -------------------------------------------------------------- //
// SETUP
// -------------------------------------------------------------- //
void setup() {
    Serial.begin(115200);
    while (!Serial);

    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1);
    }

    // Encoder Pins
    pinMode(SIGNAL_A_L, INPUT);
    pinMode(SIGNAL_B_L, INPUT);
    pinMode(SIGNAL_A_R, INPUT);
    pinMode(SIGNAL_B_R, INPUT);

    // Motor Pins
    pinMode(M_EA_L, OUTPUT); pinMode(M_I1, OUTPUT); pinMode(M_I2, OUTPUT);
    pinMode(M_EA_R, OUTPUT); pinMode(M_I3, OUTPUT); pinMode(M_I4, OUTPUT);

    // Attach Interrupts (CRITICAL FIX)
    attachInterrupt(digitalPinToInterrupt(SIGNAL_A_L), decodeEncoderTicksL, RISING);
    attachInterrupt(digitalPinToInterrupt(SIGNAL_A_R), decodeEncoderTicksR, RISING);

    calibrateGyro();
    
    // Initial State
    resetEncoders();
    last_loop_time = millis();
    currentState = STATE_FORWARD_1;
    
    delay(1000); // Pause before starting
}

// -------------------------------------------------------------- //
// LOOP
// -------------------------------------------------------------- //
void loop() {
    // 1. Calculate Delta Time
    unsigned long now = millis();
    float dt = (now - last_loop_time) / 1000.0;
    last_loop_time = now;

    // 2. Read Gyro and Update Heading
    float gx, gy, gz;
    if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(gx, gy, gz);
        // Integrate angular rate to get angle: Angle += (Rate - Offset) * Time
        current_heading += (gz - gyro_z_offset) * dt;
    }

    // 3. State Machine Logic
    switch (currentState) {

        // --- STEP 1: DRIVE FORWARD 1 METER ---
        case STATE_FORWARD_1: {
            long dist = getAvgEncoderDistance();
            
            // PID for going straight (Target Heading = 0)
            float error = 0.0 - current_heading; 
            float correction = error * kp_heading;

            setMotors(base_speed_pwm - correction, base_speed_pwm + correction);

            // Check if 1 meter reached
            if (dist >= TICKS_PER_METER) {
                stopMotors();
                delay(500); // Settle
                resetEncoders();
                currentState = STATE_TURN_180;
            }
            break;
        }

        // --- STEP 2: TURN 180 DEGREES ---
        case STATE_TURN_180: {
            // Simple P-controller for turning
            float target = 180.0;
            float error = target - current_heading;

            // Stop if we are close enough (within 2 degrees)
            if (abs(error) < 2.0) {
                stopMotors();
                delay(500);
                resetEncoders();
                currentState = STATE_FORWARD_2;
            } else {
                // Determine turn speed (minimum 80 to overcome friction)
                float turnSpeed = 2.0 * error; 
                // Clamp speed
                if(turnSpeed > 100) turnSpeed = 100;
                if(turnSpeed < -100) turnSpeed = -100;
                // Min power boost
                if(turnSpeed > 0 && turnSpeed < 60) turnSpeed = 60;
                if(turnSpeed < 0 && turnSpeed > -60) turnSpeed = -60;

                // Spin in place (Left forward, Right backward for positive turn)
                setMotors(-turnSpeed, turnSpeed);
            }
            break;
        }

        // --- STEP 3: DRIVE BACK 1 METER ---
        case STATE_FORWARD_2: {
            long dist = getAvgEncoderDistance();

            // PID for going straight (Target Heading = 180)
            float error = 180.0 - current_heading;
            float correction = error * kp_heading;

            setMotors(base_speed_pwm - correction, base_speed_pwm + correction);

            // Check if 1 meter reached
            if (dist >= TICKS_PER_METER) {
                stopMotors();
                currentState = STATE_STOP;
            }
            break;
        }

        // --- STEP 4: DONE ---
        case STATE_STOP:
            stopMotors();
            // Do nothing
            break;
            
        default:
            stopMotors();
            break;
    }
    
    // Small loop delay to keep loop rate consistent (approx 100Hz)
    delay(10);
}