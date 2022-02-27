#!/usr/bin/env python3
from pyfirmata import Arduino, SERVO, INPUT, OUTPUT
from pyfirmata.util import Iterator
from time import sleep

def rotateServo(pin, angle):
    for i in range(0, angle):
        board.digital[pin].write(i)
        sleep(0.015) 
    
if __name__ == '__main__':
    board = Arduino('COM3')
    print("Communication Successfully started")
    
    pinServo = 9
    pinRelay = 10
    pinSensor = 3
    
    
    servo = board.digital[pinServo]
    relay = board.digital[pinRelay]
    sensor = board.digital[pinSensor]
    
    servo.mode = SERVO
    relay.mode = OUTPUT
    sensor.mode = INPUT
    
    it = Iterator(board)
    it.start()

