/**
2 * @file PWM-motor-control.ino
3 * @author Joshua Marshall (joshua.marshall@queensu.ca)
4 * @brief Arduino program to drive one wheel motor through a motor driver.
5 * @version 2.0
6 * @date 2022-12-05
7 *
8 * @copyright Copyright (c) 2021-2022
9 *
10 */

int EA = 11; // Wheel PWM pin (must be a PWM pin)
int I1 = 8; // Wheel direction digital pin 1
int I2 = 9; // Wheel direction digital pin 2



 void setup()
 {
// Configure digital pins for output
pinMode(EA, OUTPUT);
pinMode(I1, OUTPUT);
pinMode(I2, OUTPUT);
}

void loop()
 {
int u; // A variable for the motor PWM command [0-255]

// Play with this code to write open loop commands to a wheel motor
for (u = 0; u <= 255; u += 5)
 {
// Select a direction
digitalWrite(I1, LOW);
digitalWrite(I2, HIGH);
// PWM command to the motor driver
analogWrite(EA, u);

// Brief delay (perhaps not necessary)
 delay(10);
 }
 }