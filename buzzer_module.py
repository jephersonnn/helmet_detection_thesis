from gpiozero import Buzzer
from time import sleep

buzzer = Buzzer(17)

def start_up():
    buzzer.on()
    sleep(0.05)
    buzzer.off()
    sleep(0.075)
    buzzer.on()
    sleep(0.075)
    buzzer.off()
    
def goodbye():
    buzzer.on()
    sleep(0.05)
    buzzer.off()
    sleep(0.075)
    buzzer.on()
    sleep(0.075)
    buzzer.off()
    sleep(0.075)
    buzzer.on()
    sleep(0.075)
    buzzer.off()
    sleep(0.075)
    buzzer.on()
    sleep(0.075)
    buzzer.off()
    
def bz_warn():
    buzzer.beep()
    
def bz_off():
    buzzer.off()
        


