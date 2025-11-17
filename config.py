ULTRASONIC_PINS = {
    "US_L": {"trigger": 5, "echo": 6},
    "US_M": {"trigger": 13, "echo": 19},
    "US_R": {"trigger": 20, "echo": 21},
}

ADC_ADDRESS = 0x48
IR_CHANNELS = {
    "IR_L": 0,
    "IR_M": 1,
    "IR_R": 2
}

# num of readings to measure temporal stability
IR_TEMPORAL_SAMPLES = 20
