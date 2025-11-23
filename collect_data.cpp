#include <SharpIR.h>

const int IR_PIN = A0;
SharpIR ir_sensor(SharpIR::GP2Y0A21YK0F, IR_PIN);

const int PW_PIN = 5;
const int US_ANALOG_PIN = A2;

void setup()
{
    Serial.begin(115200);
    delay(500);
    Serial.println("IR_raw,IR_distance_cm,US_pw_us,US_analog_raw");
}

void loop()
{
    int ir_raw = analogRead(IR_PIN);                // raw analog voltage (0–1023)
    int ir_distance_cm = ir_sensor.getDistance();   // processed distance
    unsigned long us_pw_us = pulseIn(PW_PIN, HIGH); // raw PW pulse width (µs)
    int us_analog_raw = analogRead(US_ANALOG_PIN);  // raw analog reading
    unsigned long t_ms = millis();

    Serial.print(t_ms);
    Serial.print(",");
    Serial.print(ir_raw);
    Serial.print(",");
    Serial.print(ir_distance_cm);
    Serial.print(",");
    Serial.print(us_pw_us);
    Serial.print(",");
    Serial.println(us_analog_raw);

    delay(100);
}
