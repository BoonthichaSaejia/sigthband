
import cv2
import depthai as dai
import numpy as np
from scipy.signal import find_peaks
import math
import time
from pathlib import Path
import sys
import RPi.GPIO as GPIO
import time

# vibration motor
left = 23
center = 24
right  = 25
GPIO.setmode(GPIO.BCM)
GPIO.setup(left, GPIO.OUT)
GPIO.setup(center, GPIO.OUT)
GPIO.setup(right, GPIO.OUT)

def findPeak(list_depth, n_iter=10):
    arr_depth = np.array(list_depth)
    #width=35 is where the model cannot detect any peaks while it cannot see anything
    peaks, _ = find_peaks(arr_depth, prominence=1, width=37)
    if len(peaks) == 0:
        return 0
    else:
        idx_max_peak = np.max(np.array(peaks))
        return arr_depth[idx_max_peak]
 
# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = True
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True
 
# Create pipeline
pipeline = dai.Pipeline()
 
# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
xout = pipeline.create(dai.node.XLinkOut)
 
xout.setStreamName("disparity")
 
# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
 
# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)
 
# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout.input)
 
# Connect to device and start pipeline
with dai.Device(pipeline) as device:
 
    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    count = 0
    top_head = 100
    top_body = 200
    top_leg = 300
    FAR_DISTANCE = 500
    BASELINE = 7.0
    HFOV = 72
    width_section = 9
 
    while True:
        start_time = time.time()
        count = count+1
        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getFrame()
        # Normalization for better visualization
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        for i in range(1,width_section):
            frame = cv2.line(frame, (int(640/width_section*i), 0),(int(640/width_section*i), 400), (255, 0, 0), thickness=1)
        
        frame = cv2.line(frame, (0, top_head),(640, top_head), (255, 0, 0), thickness=1)
        frame = cv2.line(frame, (0, top_body),(640, top_body), (255, 0, 0), thickness=1)
        frame = cv2.line(frame, (0, top_leg), (640, top_leg),  (255, 0, 0), thickness=1)
  
        warning_section = np.zeros((3, width_section))
        width = 640
        threshold = 0
        n_iter = 1
        focal_length_in_pixels = width * 0.5 / math.tan(HFOV * 0.5 * math.pi / 180)
 
        for i in range(width_section):
            lst_depth = []
            rep_depth = 0
            for j in range(int(width/width_section*i),int(width/width_section*(i+1))):
                for k in range(top_head,top_body):
                    if (frame[k][j] > threshold ):
                        lst_depth.append(frame[k][j])
            if len(lst_depth)==0:
                rep_depth = FAR_DISTANCE
            else:
                result = findPeak(lst_depth, n_iter)
                if result == 0:
                    rep_depth = FAR_DISTANCE
                else:
                    rep_depth = int(focal_length_in_pixels * BASELINE / result)
            warning_section[0][i] = rep_depth
 
            lst_depth = []
            rep_depth = 0
            for j in range(int(width / width_section * i), int(width / width_section * (i + 1))):
                for k in range(top_body,top_leg):
                    if (frame[k][j] > threshold):
                        lst_depth.append(frame[k][j])
            if len(lst_depth) == 0:
                rep_depth = FAR_DISTANCE
            else:
                result = findPeak(lst_depth, n_iter)
                if result == 0:
                    rep_depth = FAR_DISTANCE
                else:
                    rep_depth = int(focal_length_in_pixels * BASELINE / result)
            warning_section[1][i] = rep_depth
 
            lst_depth = []
            rep_depth = 0
            for j in range(int(width / width_section * i), int(width / width_section * (i + 1))):
                for k in range(top_leg,400):
                    if (frame[k][j] > threshold ):
                        lst_depth.append(frame[k][j])
            if len(lst_depth) == 0:
                rep_depth = FAR_DISTANCE
            else:
                result = findPeak(lst_depth, n_iter)
                if result == 0:
                    rep_depth = FAR_DISTANCE
                else:
                    rep_depth = int(focal_length_in_pixels * BASELINE / result)
            warning_section[2][i] = rep_depth

        def labelLocation(j):
            if j<=1:
                return 'left'
            elif j>=7:
                return 'right'
            elif j>=3 and j<=5:
                return 'center'
            else:
                return 'void'
    
        def vibrate(sensor, serious_level):
            if sensor == 'void':
                pass
            else:
                if sensor == 'left':
                    channel = left
                elif sensor == 'right':
                    channel = right
                elif sensor == 'center':
                    channel = center
                GPIO.output(channel, GPIO.HIGH)
                if serious_level == 1:
                    time.sleep(0.2)
                elif serious_level == 2:
                    time.sleep(0.5)
                else :
                    time.sleep(0)
                GPIO.output(channel, GPIO.LOW)
        
        for i in range(3):
            max_num=np.ones(width_section)*(-1)
            for j in range(width_section):
                if warning_section[2-i][8-j] > max_num[j]:
                    max_num[j] = warning_section[i][8-j]
        for i in range(width_section):
            if max_num[i] >= 60 and (max_num[2-i]<=160):
                vibrate(labelLocation(i),1)
            elif (max_num[i] <60) and (max_num[2-i]>=0):
                vibrate(labelLocation(i),2)
            print(labelLocation(i),max_num[i])
 
        whiteblankimage = 255 * np.ones(shape=[400, 640, 3], dtype=np.uint8)
        for i in range(3):
            for j in range(width_section):
                if warning_section[i][j] >= 400:
                    cv2.rectangle(whiteblankimage, pt1=(int(640/width_section)*j, (100*i)+100), pt2=(int(640/width_section)*(j+1), (100*i)+200), color=(224, 253, 222),thickness=-1)
                elif warning_section[i][j] >= 140:
                    cv2.rectangle(whiteblankimage, pt1=(int(640/width_section)*j, (100*i)+100), pt2=(int(640/width_section)*(j+1), (100*i)+200), color=(222, 247, 252), thickness=-1)
                elif warning_section[i][j] >= 60:
                    cv2.rectangle(whiteblankimage, pt1=(int(640/width_section)*j, (100*i)+100), pt2=(int(640/width_section)*(j+1), (100*i)+200), color=(211, 223, 255), thickness=-1)
                else:
                    cv2.rectangle(whiteblankimage, pt1=(int(640/width_section)*j, (100*i)+100), pt2=(int(640/width_section)*(j+1), (100*i)+200), color=(162, 154, 255), thickness=-1)
        cv2.imshow("warning_color", whiteblankimage)
        print("This round is %s seconds"%(time.time()-start_time))
        if cv2.waitKey(1) == ord('q'):
            GPIO.cleanup()
            break

