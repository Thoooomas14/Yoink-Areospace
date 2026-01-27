/**
 * @file IMU-measurements.ino
 * @author Joshua Marshall (joshua.marshall@queensu.ca), Thomas Sears (thomas.sears@queensu.ca)
 * @brief Arduino program to read the LSM6DS3 IMU on the Arduino UNO WiFi Rev2.
 * @version 1.1
 * @date 2022-11-30
 *
 * @copyright Copyright (c) 2022
 *
 */

// Need this library installed to access the UNO Wifi Rev2 board's IMU
// For more details, look here: https://www.arduino.cc/reference/en/libraries/arduino_lsm6ds3/

#include <Arduino_LSM6DS3.h>

// Variables to store angular rates from the gyro [degrees/s]
float omega_x, omega_y, omega_z;

// Variables to store accelerations [g's]
float a_x, a_y, a_z;

// Variables to store sample rates from sensor [Hz]
float a_f, g_f;

double X_offset; 
double Y_offset; 
double Z_offset; 

void setup()
{
    // Open the serial port at 115200 bps
    Serial.begin(115200);

    // Wait for serial connection before starting
    while (!Serial)
    {
        delay(10);
    }

    Serial.println();

    // Check that the board is initialized
    if (!IMU.begin())
    {
        // Print an error message if the IMU is not ready
        Serial.print("Failed to initialize IMU :(");
        Serial.print("\n");
        while (1)
        {
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
    for(int i=0; i<1000; i++){
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

}

void loop()
{
    // Timing in the loop is controlled by the IMU reporting when
    // it is ready for another measurement.
    // The accelerometer and gyroscope output at the same rate and
    // will give us their measurements at a steady frequency.

    // Read from the accelerometer
    if (IMU.accelerationAvailable())
    {
        IMU.readAcceleration(a_x, a_y, a_z);

        // Print the accelerometer measurements to the Serial Monitor
        Serial.print(a_x);
        Serial.print("\t");
        Serial.print(a_y);
        Serial.print("\t");
        Serial.print(a_z);
        Serial.print(" g\t\t");
    }

    // Read from the gyroscope
    if (IMU.gyroscopeAvailable())
    {
        IMU.readGyroscope(omega_x, omega_y, omega_z);

        omega_x = omega_x + X_offset;
        omega_y = omega_y + Y_offset;
        omega_z = omega_z + Z_offset;
        // Print the gyroscope measurements to the Serial Monitor
        Serial.print(omega_x);
        Serial.print("\t");
        Serial.print(omega_y);
        Serial.print("\t");
        Serial.print(omega_z);
        Serial.print(" deg/s\n");
    }
    delay(100);
}