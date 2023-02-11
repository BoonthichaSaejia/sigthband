import RPi.GPIO as GPIO
import time

left = 23
center = 24
right  = 25
GPIO.setmode(GPIO.BCM)
GPIO.setup(left, GPIO.OUT)
GPIO.setup(center, GPIO.OUT)
GPIO.setup(right, GPIO.OUT)


def vibrate(sensor):
    GPIO.output(sensor, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(sensor, GPIO.LOW)
    print(sensor)

for i in range(10):    
    vibrate(left)
    vibrate(center)
    vibrate(right)
GPIO.cleanup()
