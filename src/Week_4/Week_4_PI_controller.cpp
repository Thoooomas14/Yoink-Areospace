#include <Arduino.h>
#include <Arduino_LSM6DS3.h>

// -------------------------------------------------------------- //
// GLOBAL VARIABLES (IMU / PID state)
float integral = 0;
float lastError = 0;

// Variables to store angular rates from the gyro [degrees/s]
float omega_x = 0.0f, omega_y = 0.0f, omega_z = 0.0f;

// Variables to store accelerations [g's]
float a_x = 0.0f, a_y = 0.0f, a_z = 0.0f;

// Variables to store sample rates from sensor [Hz]
float a_f = 0.0f, g_f = 0.0f;

double X_offset = 0.0;
double Y_offset = 0.0;
double Z_offset = 0.0;

float Kp_speed = 5.0f;
float Ki_speed = 0.0f;
float Kd_speed = 0.0f;


float current_heading = 0.0; // Current robot angle in degrees
float current_position = 0.0; // Current robot position in meters

float targetSpeed = 0.0f;
float targetRotation = 0.0f;

unsigned long previousMicros = 0;

int state = 0;
int count = 0;
int DELAY = 0;
// -------------------------------------------------------------- //
// Encoder / wheel parameters (self-contained)
// --- ENCODER PINS ---
const uint8_t SIGNAL_A_L = 2;
const uint8_t SIGNAL_B_L = 3;
const uint8_t SIGNAL_A_R = 11;
const uint8_t SIGNAL_B_R = 12;

// Encoder ticks per motor revolution (TPR)
const int TPR = 3000;

// Wheel parameter for converting angular rate to linear speed (m)
double p = 0.0625;

// Counters to keep track of encoder ticks [integer]
volatile long encoder_ticks_L = 0;
volatile long encoder_ticks_R = 0;

// Variables to store estimated angular rate [rad/s]
double omega_L = 0.0;
double omega_R = 0.0;

// --- INTERRUPT SERVICE ROUTINES ---
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

// Use local constants for motor pins to avoid linker conflicts while matching wiring.
static const uint8_t M_EA_R = 5; // Left motor PWM
static const uint8_t M_I3 = 6;
static const uint8_t M_I4 = 7;

static const uint8_t M_EA_L = 10; // Right motor PWM
static const uint8_t M_I1 = 8;
static const uint8_t M_I2 = 9;

// Separate PID state for left and right wheels
static float integral_L = 0.0f;
static float lastError_L = 0.0f;
static float integral_R = 0.0f;
static float lastError_R = 0.0f;

const float TRACK_WIDTH = 0.2775f; // Track width in meters

// PID helper that uses external state references
float PIDControlState(float sensorValue, float setpoint, float Kp, float Ki, float Kd, float &integralRef, float &lastErrorRef, float timeStep) {
    float error = setpoint - sensorValue;
    integralRef += error * timeStep;
    float derivative = (error - lastErrorRef) / timeStep;
    float output = Kp * error + Ki * integralRef + Kd * derivative;
    lastErrorRef = error;
    return output;
}

// Function to read speed from encoder
float readSpeed() {
    // Compute/update wheel angular rates and return average linear speed [m/s]
    static float lastSpeed = 0.0f;
    static unsigned long last_time = 0;
    const unsigned long T_local = 50; // ms sampling interval (reduced for better PID response)

    unsigned long now = millis();
    if ((now - last_time) >= T_local) {
        double dt = (double)(now - last_time); // ms

        // Safely capture and reset encoder tick counters
        noInterrupts();
        long ticksL = encoder_ticks_L;
        long ticksR = encoder_ticks_R;
        encoder_ticks_L = 0;
        encoder_ticks_R = 0;
        interrupts();

        // Estimate rotational speeds [rad/s]
        double local_omega_L = 2.0 * PI * ((double)ticksL / (double)TPR) * 1000.0 / dt;
        double local_omega_R = 2.0 * PI * ((double)ticksR / (double)TPR) * 1000.0 / dt;

        // Update the extern globals (so other modules can read them)
        omega_L = local_omega_L;
        omega_R = local_omega_R;

        // Linear speed [m/s] (average of both wheels)
        float speed = 0.5f * (float)p * (float)(local_omega_L + local_omega_R);

        last_time = now;
        lastSpeed = speed;
        return speed;
    }

    // If not time to recompute yet, return last computed value
    return lastSpeed;
}

// Function to read rotation rate from IMU
float readRotationRate() {
    // Read IMU gyroscope (if available) and update global omega_x/y/z with offsets
    if (IMU.gyroscopeAvailable()) {
        float gx = 0.0f, gy = 0.0f, gz = 0.0f;
        IMU.readGyroscope(gx, gy, gz);

        // Apply previously computed offsets (from setup())
        omega_x = gx + (float)X_offset;
        omega_y = gy + (float)Y_offset;
        omega_z = gz + (float)Z_offset;

        return omega_z;
    }

    // No new sample available — return last-known z-rate
    return omega_z;
}

void setup() {
    // Open the serial port at 115200 bps
    Serial.begin(57600);

    // Wait for serial connection before starting
    while (!Serial) {
        delay(10);
    }

    Serial.println();

    // Check that the board is initialized
    if (!IMU.begin()) {
        // Print an error message if the IMU is not ready
        Serial.print("Failed to initialize IMU :(");
        Serial.print("\n");
        while (1) {
            delay(10);
        }
    }

    // Read the sample rate of the accelerometer and gyroscope
    a_f = IMU.accelerationSampleRate();
    g_f = IMU.gyroscopeSampleRate();

    // Print these values to the serial window
    Serial.print("Accelerometer sample rate: ");
    Serial.println(a_f);
    Serial.print("Gyroscope sample rate: ");
    Serial.println(g_f);

    float ox = 0.0f;
    float oy = 0.0f;
    float oz = 0.0f;
    double avgx = 0.0;
    double avgy = 0.0;
    double avgz = 0.0;
    for (int i = 0; i < 1000; i++) {
        IMU.readGyroscope(ox, oy, oz);
        avgx += ox;
        avgy += oy;
        avgz += oz;
        delay(1);
    }
    // compute negative average (offset) using floating point division
    X_offset = -avgx / 1000.0;
    Y_offset = -avgy / 1000.0;
    Z_offset = -avgz / 1000.0;

    // Debug: print computed offsets
    Serial.print("Gyro offsets (deg/s): ");
    Serial.print(X_offset);
    Serial.print(", ");
    Serial.print(Y_offset);
    Serial.print(", ");
    Serial.println(Z_offset);

    // Initialize encoder pins
    pinMode(SIGNAL_A_L, INPUT);
    pinMode(SIGNAL_B_L, INPUT);
    pinMode(SIGNAL_A_R, INPUT);
    pinMode(SIGNAL_B_R, INPUT);

    // Attach interrupts for encoders
    attachInterrupt(digitalPinToInterrupt(SIGNAL_A_L), decodeEncoderTicksL, RISING);
    attachInterrupt(digitalPinToInterrupt(SIGNAL_A_R), decodeEncoderTicksR, RISING);

    // Initialize motor pins (use local pin names to avoid symbol collisions)
    pinMode(M_EA_L, OUTPUT);
    pinMode(M_I1, OUTPUT);
    pinMode(M_I2, OUTPUT);

    pinMode(M_EA_R, OUTPUT);
    pinMode(M_I3, OUTPUT);
    pinMode(M_I4, OUTPUT);

    // Start stopped (forward direction pins default)
    digitalWrite(M_I1, LOW);
    digitalWrite(M_I2, HIGH);
    analogWrite(M_EA_L, 0);

    digitalWrite(M_I3, LOW);
    digitalWrite(M_I4, HIGH);
    analogWrite(M_EA_R, 0);

    delay(1000); // Wait a moment before starting
    
    // No shared t_last here; readSpeed uses its own internal timing
}

void loop() {
    unsigned long currentMicros = micros();
    // Guard against wrap-around or cleared counter, but mostly speed.
    // Use roughly 1ms minimum check if you want, but simply checking for 0 dt is faster.
    // To properly calculate dt using micros:
    if (currentMicros <= previousMicros) {
        // Just in case of weirdness or super fast loop
        previousMicros = currentMicros;
        return; 
    }
    float timestep = (currentMicros - previousMicros) / 1000000.0f;
    previousMicros = currentMicros;
    // Read sensors
    float speed = readSpeed(); // m/s
    float rotationRate = readRotationRate(); // deg/s (gyro z)

    current_heading = fmod(current_heading + rotationRate * timestep, 360.0f); // Update heading
    current_position += speed * timestep; // Update position

    switch (state)
    {
    case 0:
        targetSpeed = 0.5f;
        targetRotation = 0.0f;
        if (current_position >= 1.0f) {
            current_position = 0.0f; // reset for next leg
            state = 1;
            DELAY = 1;
            integral_L = 0.0f;
            lastError_L = 0.0f;
            integral_R = 0.0f;
            lastError_R = 0.0f;
        }
        break;
    case 1:
        targetRotation = 0.0f;
        targetSpeed = 0.0f;
        if (DELAY % 100 == 0) {
            state = 2;
            current_position = 0.0f;
            current_heading = 0.0f;
            DELAY = 0;
            integral_L = 0.0f;
            lastError_L = 0.0f;
            integral_R = 0.0f;
            lastError_R = 0.0f;
        }
        else {
            DELAY ++;
        }
    break;
    case 2:
        targetSpeed = 0.0f;
        targetRotation = 5.0f;
        if (abs(abs(current_heading) - 160.0f) < 5.0f) { // within 5 degrees of target
            current_heading = 0.0f; // snap to exact
            current_position = 0.0f; // reset for next leg
            state = 0;
        }
        break;
    default:
        break;
    }

    // Calculate measured speeds for each wheel [m/s]
    float speed_L = (float)omega_L * (float)p;
    float speed_R = (float)omega_R * (float)p;

    // Calculate target speeds for each wheel (Unicycle model)
    // Assumes targetRotation is in rad/s.
    float target_speed_L = targetSpeed - (targetRotation * TRACK_WIDTH / 2.0f);
    float target_speed_R = targetSpeed + (targetRotation * TRACK_WIDTH / 2.0f);

    // Run PID for each wheel
    float control_L = PIDControlState(speed_L, target_speed_L, Kp_speed, Ki_speed, Kd_speed, integral_L, lastError_L, timestep);
    float control_R = PIDControlState(speed_R, target_speed_R, Kp_speed, Ki_speed, Kd_speed, integral_R, lastError_R, timestep);

    if (count % 10 == 0) {
        Serial.print("Error Left: "); Serial.print(target_speed_L - speed_L); Serial.print(" Error Right: "); Serial.print(target_speed_R - speed_R);
        Serial.print("Actual Left: "); Serial.print(speed_L); Serial.print(" Actual Right: "); Serial.println(speed_R);
        // Serial.print("Overall Target: "); Serial.print(targetSpeed); Serial.print(", "); Serial.print(speed); Serial.print(" | ");
        // Serial.print("Target Left: "); Serial.print(target_speed_L); Serial.print(" Actual Left: "); Serial.print(speed_L); Serial.print(" | ");
        // Serial.print("Target Right: "); Serial.print(target_speed_R); Serial.print(" Actual Right: "); Serial.println(speed_R); Serial.print(" | ");
        // Serial.print("Position: "); Serial.print(current_position); Serial.print(" Heading: "); Serial.print(current_heading); Serial.print(" | ");
        // Serial.print("Control Left: "); Serial.print(control_L); Serial.print(" Control Right: "); Serial.println(control_R); Serial.print(" | ");
        // Serial.print("State: "); Serial.println(state);
        count = 0;
    }

    // Map PID outputs to motor PWM commands
    float speedToPWM = 100.0f;
    
    float leftPWMf = control_L * speedToPWM;
    float rightPWMf = control_R * speedToPWM;

    // Set motor directions based on sign of PWM
    // LEFT MOTOR
    if (round(leftPWMf) >= 0) {
        digitalWrite(M_I1, LOW);
        digitalWrite(M_I2, HIGH);
        analogWrite(M_EA_L, constrain((int)round(leftPWMf), 0, 255));
    } else {
        digitalWrite(M_I1, HIGH);
        digitalWrite(M_I2, LOW);
        analogWrite(M_EA_L, constrain((int)round(-leftPWMf), 0, 255));
    }

    // RIGHT MOTOR
    if (round(rightPWMf) >= 0) {
        digitalWrite(M_I3, LOW);
        digitalWrite(M_I4, HIGH);
        analogWrite(M_EA_R, constrain((int)round(rightPWMf), 0, 255));
    } else {
        digitalWrite(M_I3, HIGH);
        digitalWrite(M_I4, LOW);
        analogWrite(M_EA_R, constrain((int)round(-rightPWMf), 0, 255));
    }

    delay(100);
    count ++;
}