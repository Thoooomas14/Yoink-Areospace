/**
 * @file motor-angular-rate.ino
 * @brief Arduino program to estimate motor speed from encoder (Left and Right).
 */

// --- LEFT MOTOR PINS ---
// Wheel PWM pin
int EA_L = 5;
// Wheel direction digital pins
int I1 = 6;
int I2 = 7;

// --- RIGHT MOTOR PINS ---
// Wheel PWM pin (User requested EA2=10)
int EA_R = 10;
// Wheel direction digital pins (User requested I3=9, I4=8)
int I3 = 8;
int I4 = 9;

// Motor PWM command variables [0-255]
byte u_L = 0;
byte u_R = 0;

// --- ENCODER PINS ---
// Left wheel encoder
const byte SIGNAL_A_L = 2;
const byte SIGNAL_B_L = 3;

// Right wheel encoder (User requested 11 and 12)
const byte SIGNAL_A_R = 11;
const byte SIGNAL_B_R = 12;

// Encoder ticks per (motor) revolution (TPR)
const int TPR = 3000;

// Wheel radius [m]
const double RHO = 0.0625;

// Counters to keep track of encoder ticks [integer]
volatile long encoder_ticks_L = 0;
volatile long encoder_ticks_R = 0;

// Variables to store estimated angular rate [rad/s]
double omega_L = 0.0;
double omega_R = 0.0;

// Sampling interval for measurements in milliseconds
const int T = 1000;

// Counters for milliseconds during interval
long t_now = 0;
long t_last = 0;


double p = 0.0625;
// --- INTERRUPT SERVICE ROUTINES ---

// Called when SIGNAL_A_L goes HIGH
void decodeEncoderTicksL()
{
    if (digitalRead(SIGNAL_B_L) == LOW)
    {
        encoder_ticks_L--;
    }
    else
    {
        encoder_ticks_L++;
    }
}

// Called when SIGNAL_A_R goes HIGH
void decodeEncoderTicksR()
{
    if (digitalRead(SIGNAL_B_R) == LOW)
    {
        // Assuming same polarity logic as left; swap ++/-- if it counts backwards
        encoder_ticks_R++; 
    }
    else
    {
        encoder_ticks_R--;
    }
}

void setup()
{
    // Open the serial port
    Serial.begin(9600);

    // --- SETUP LEFT MOTOR ---
    pinMode(EA_L, OUTPUT);
    pinMode(I1, OUTPUT);
    pinMode(I2, OUTPUT);

    // --- SETUP RIGHT MOTOR ---
    pinMode(EA_R, OUTPUT);
    pinMode(I3, OUTPUT);
    pinMode(I4, OUTPUT);

    // --- SETUP LEFT ENCODER ---
    pinMode(SIGNAL_A_L, INPUT);
    pinMode(SIGNAL_B_L, INPUT);
    attachInterrupt(digitalPinToInterrupt(SIGNAL_A_L), decodeEncoderTicksL, RISING);

    // --- SETUP RIGHT ENCODER ---
    pinMode(SIGNAL_A_R, INPUT);
    pinMode(SIGNAL_B_R, INPUT);
    // Note: Pin 11 interrupt works on Nano Every/MegaAVR boards
    attachInterrupt(digitalPinToInterrupt(SIGNAL_A_R), decodeEncoderTicksR, RISING);

    Serial.println("Program initialized (Left & Right).");
}

const double ELL = 0.2775;

double compute_vehicle_rate(double v_L, double v_R)
{
    double omega;
    omega = 1.0 / ELL * (v_R - v_L);
    return omega;
}
void loop()
{
    // Get the elapsed time [ms]
    t_now = millis();

    if (t_now - t_last >= T)
    {
        // 1. Calculate Time Step
        double dt = (double)(t_now - t_last);

        // 2. Estimate rotational speeds [rad/s]
        omega_L = 2.0 * PI * ((double)encoder_ticks_L / (double)TPR) * 1000.0 / dt;
        omega_R = 2.0 * PI * ((double)encoder_ticks_R / (double)TPR) * 1000.0 / dt;

        // 3. Print to Serial Monitor
        Serial.print("L_Ticks: "); Serial.print(encoder_ticks_L);
        Serial.print("\t L_Speed: "); Serial.print(omega_L);
        Serial.print(" rad/s");
        Serial.print(omega_L*p);
        Serial.print(" m/s");
        
        Serial.print("\t | \t"); // Separator

        Serial.print("R_Ticks: "); Serial.print(encoder_ticks_R);
        Serial.print("\t R_Speed: "); Serial.print(omega_R);
        Serial.print(" rad/s");
        Serial.print(omega_R*p);
        Serial.print(" m/s");

        Serial.print("\t | \t"); // Separator

        Serial.print("Average Speed: ");
        Serial.print(0.5*p*(omega_R+omega_L));

        Serial.print("\t | \t"); // Separator
        Serial.print("Turning Rate");
        Serial.print(compute_vehicle_rate(omega_L*p, omega_R*p));
        Serial.println();

        // 4. Reset
        t_last = t_now;
        encoder_ticks_L = 0;
        encoder_ticks_R = 0;
    }

    // --- DRIVE MOTORS ---
    
    // Set PWM commands
    u_L = 128; 
    u_R = 128;

    // LEFT MOTOR Direction
    digitalWrite(I1, LOW);
    digitalWrite(I2, HIGH);
    analogWrite(EA_L, u_L);

    // RIGHT MOTOR Direction
    digitalWrite(I3, LOW); 
    digitalWrite(I4, HIGH);
    analogWrite(EA_R, u_R);
}