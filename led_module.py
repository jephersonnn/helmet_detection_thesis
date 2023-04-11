from gpiozero import RGBLED
from colorzero import Color
from time import sleep

led = RGBLED(red = 17, green = 27, blue = 22)
    
#led.on()

def h_off():
    led.blink(on_time=1, off_time=1, on_color=(0,1,1), off_color=(1,1,1), background=True)
    
def h_on():
    led.color = Color(1,0,1)
    
def h_neutral():
    led.pulse(fade_in_time=0.7, fade_out_time=0.7, on_color=(1,0,1), off_color=(1,1,1), background=True)
    
def led_start():
    led.color = Color(1,1,0)
    sleep(0.1)
    led.color = Color(1,0,1)
    sleep(0.1)
    led.color = Color(0,1,1)
    sleep(0.1)
    led.color = Color(1,0,1)
    sleep(0.1)
    led.pulse(fade_in_time=0.5, fade_out_time=0.5, on_color=(1,1,0), off_color=(1,1,1), background=True)
    

def goodbye_led():
    led.on()
    
    
    