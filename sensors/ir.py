import time
import board
import busio
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_ads1x15.ads1115 as ADS
from config import IR_TEMPORAL_SAMPLES

class IRSensor:
    def __init__(self, i2c, channel):
        ads = ADS.ADS1115(i2c)
        self.chan = AnalogIn(ads, channel)

    def read(self):
        """Returns (voltage, temporal_stability)"""
        readings = []
        for _ in range(IR_TEMPORAL_SAMPLES):
            readings.append(self.chan.voltage)
            time.sleep(0.002)

        voltage = sum(readings) / len(readings)
        variance = sum((r - voltage)**2 for r in readings) / len(readings)

        return voltage, variance
