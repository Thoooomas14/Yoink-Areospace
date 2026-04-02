int pinI1=8;//define I1 port
int pinI2=9;//define I2 port
int speedpin=5;//define EA(PWM speed regulation)port

int pinI3=6;//define I3 port
int pinI4=7;//define I4 port
int speed=10;

void setup()
{
 pinMode(pinI1,OUTPUT);//define this port as output
 pinMode(pinI2,OUTPUT);
 pinMode(speedpin,OUTPUT);
 pinMode(pinI3,OUTPUT);//define this port as output
 pinMode(pinI4,OUTPUT);
 pinMode(speed,OUTPUT);
}
void loop()
{
 analogWrite(speedpin,128);//input a value to set the speed
 analogWrite(speed,255);
 delay(2000);
 digitalWrite(pinI1,LOW);// DC motor rotates clockwise
 digitalWrite(pinI2,HIGH);
 digitalWrite(pinI3,LOW);// DC motor rotates clockwise
 digitalWrite(pinI4,HIGH);
 analogWrite(speedpin,128);
 analogWrite(speed,255);
 delay(2000);

}