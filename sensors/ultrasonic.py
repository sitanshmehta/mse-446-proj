import RPi.GPIO as GPIO
import time

class UltrasonicSensor:
    def __init__(self, trigger_pin, echo_pin):
        self.trigger = trigger_pin
        self.echo = echo_pin

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trigger, GPIO.OUT)
        GPIO.setup(self.echo, GPIO.IN)

    def _send_pulse(self):
        GPIO.output(self.trigger, True)
        time.sleep(0.00001) 
        GPIO.output(self.trigger, False)

    def read(self):
        """Returns (tof_us, echo_amplitude, echo_width_us)"""

        self._send_pulse()

        # measure ToF 
        start = time.time()
        while GPIO.input(self.echo) == 0:
            if time.time() - start > 0.02:
                return None

        pulse_start = time.time()

        while GPIO.input(self.echo) == 1:
            if time.time() - pulse_start > 0.02:
                break

        pulse_end = time.time()

        tof = (pulse_end - pulse_start) * 1e6
        echo_width = tof
        amplitude = echo_width / 1000  

        return tof, amplitude, echo_width
