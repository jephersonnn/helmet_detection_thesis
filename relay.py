import RPi.GPIO as GPIO
from time import sleep

#1 low
#0 high

def relay_boot():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(14, GPIO.OUT)
    GPIO.output(14, 1)

def unlock():
    GPIO.output(14, 0)
    
def lock():
    GPIO.output(14, 1)