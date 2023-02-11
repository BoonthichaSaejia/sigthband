
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
import pandas as pd

f = open('mydata2.txt',"a")
f.write(str('___'))  
f.close()  
# vibration motor
left = 23
center = 24
right  = 25
GPIO.setmode(GPIO.BCM)
GPIO.setup(left, GPIO.OUT)
GPIO.setup(center, GPIO.OUT)
GPIO.setup(right, GPIO.OUT)

# Get argument first
nnPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
if 1 < len(sys.argv):
    if arg == "yolo4":
        nnPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    else:
        nnPath = arg
else:
    print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")
    
if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# tiny yolo v4 label texts
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

syncNN = True
 
# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = True
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True
 
# Create pipeline
pipeline = dai.Pipeline()
 
# Define sources 
monoLeft         = pipeline.create(dai.node.MonoCamera)
monoRight        = pipeline.create(dai.node.MonoCamera)
depth            = pipeline.create(dai.node.StereoDepth)
camRgb           = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
# Define outputs
depthout = pipeline.create(dai.node.XLinkOut)
xoutRgb  = pipeline.create(dai.node.XLinkOut)
nnOut    = pipeline.create(dai.node.XLinkOut)
 
depthout.setStreamName("disparity")
xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")
 
# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
 
# Depth specific setting
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)
# Color camera specific setting
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)
# Network specific settings
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)
 
# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(depthout.input)
camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    
    # Output queue will be used to get the disparity frames from the outputs defined above
    q    = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    qRgb = device.getOutputQueue(name="rgb"      , maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn"       , maxSize=4, blocking=False)
    
    FAR_DISTANCE = 500
    BASELINE = 7.0
    HFOV = 72
    width_section = 3
    
    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)
    
    def IOU(box1,box2):
        x1, y1, x2, y2 = box1.xmin, box1.ymin, box1.xmax, box1.ymax
        x3, y3, x4, y4 = box2.xmin, box1.ymin, box1.xmax, box1.ymax
        x_inter1 = max(x1,x3)
        y_inter1 = max(y1,y3)
        x_inter2 = min(x2,x4)
        y_inter2 = min(y2,y4)
        width_inter  = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        width_box1  = abs(x2-x1)
        height_box1 = abs(y2-y1)
        width_box2  = abs(x4-x3)
        height_box2 = abs(y4-y3)
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        area_union= area_box1 + area_box2 - area_inter
        iou = area_inter / area_union
        return iou

    
    def NMS(boxes, conf_threshold=0.7, iou_threshold=0.4):
        bbox_list_thresholded = []
        bbox_list_new = []
        if len(boxes) == 0:
            return []
        boxes_sorted = sorted(boxes, reverse=True, key = lambda x:x.confidence)
        for box in boxes_sorted:
            if box.confidence > conf_threshold:
                bbox_list_thresholded.append(box)
            else:
                pass
        while len(bbox_list_thresholded) > 0:
            current_box = bbox_list_thresholded.pop(0)
            bbox_list_new.append(current_box)
            for box in bbox_list_thresholded:
                if current_box.label == box.label:
                    iou = IOU(current_box,box)
                    if iou > iou_threshold:
                        bbox_list_thresholded.remove(box)
        return bbox_list_new
    
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    
    def labelLocation(ymax,ymin):
        if ymin+(ymax-ymin)/2 < 138:
            return 'left'
        elif ymin+(ymax-ymin)/2 > 276:
            return 'right'
        else:
            return 'center'
    
    def vibrate(sensor):
        if sensor == 'left':
            channel = left
        elif sensor == 'right':
            channel = right
        elif sensor == 'center':
            channel = center
        GPIO.output(channel, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(channel, GPIO.LOW)
    def getDepth(disp):
        if disp!=0:
            focal_length_in_pixels = 400 * 0.5 / math.tan(HFOV * 0.5 * math.pi / 180)
            return int(focal_length_in_pixels * BASELINE / disp)
        else:
            return 500
    def getRealDepth(xmin,ymin,xmax,ymax,depth):
        depth_arr = np.array(depth[ymin:ymax,xmin:xmax])
        if (ymax-ymin) < 250:
            med = np.percentile(depth_arr,20)
        else:
            med = np.percentile(depth_arr, 45)
        real_depth = getDepth(med)
        return real_depth
                 
    def getWarning(xmin,ymin,xmax,ymax,depth):
        warning = 'Clear'
        depth_arr = np.array(depth[ymin:ymax,xmin:xmax])
        if (ymax-ymin) < 250:
            med = np.percentile(depth_arr,20)
        else:
            med = np.percentile(depth_arr, 45)
        real_depth = getDepth(med)
        if real_depth < 60:
            warning = 'Dangerous'
        elif 60<=real_depth<150:
            warning = 'Very Close'
        elif 150<=real_depth<300:
            warning = 'Close'
        elif 300<=real_depth<500:
            warning = 'Fine'
        return warning
    
    def displayFrame(name, frame,depth):
        color = (255, 0, 0)
        #chosen_detections = detections
        chosen_detections = NMS(detections, 0.7, 0.3)
        for detection in chosen_detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255))
            #cv2.putText(frame, f"{(detection.confidence)}", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255))
            cv2.putText(frame, labelLocation(bbox[2],bbox[0]), (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0))
            cv2.putText(frame, getWarning(bbox[0],bbox[1],bbox[2],bbox[3],depth), (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0))
            cv2.line(frame, (416//3, 0),(416//3, 416), (0, 0, 0), thickness=1)
            cv2.line(frame, (416//3*2, 0),(416//3*2, 416), (0, 0, 0), thickness=1)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            new_row = {'label':labelMap[detection.label],'location':labelLocation(bbox[2],bbox[0]),'depth':getRealDepth(bbox[0],bbox[1],bbox[2],bbox[3],depth),'range':getWarning(bbox[0],bbox[1],bbox[2],bbox[3],depth)}
            f = open('mydata2.txt',"a")
            f.write(str(new_row)+'\n')         
            f.close()                                                                                                                                                                    
            print("we detected that", labelMap[detection.label], "is", getWarning(bbox[0],bbox[1],bbox[2],bbox[3],depth), "to you.")
            cv2.imshow('objectdetection',frame)
            #vibrate(labelLocation(bbox[2],bbox[0]))

    while True:
        
        start_time = time.time()
        
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
            inDisparity = q.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()
            inDisparity = g.tryGet()
        

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
            cv2.imshow("rgb", inRgb.getCvFrame())

        if inDet is not None:
            detections = inDet.detections
            counter += 1
        
        if inDisparity is not None:
            depth_frame = inDisparity.getFrame()
            depth_frame = (depth_frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
            cropped = depth_frame[65:360,180:475]
            depth_scaled  = cv2.resize(cropped, None, fx= 1.41, fy= 1.41, interpolation= cv2.INTER_LINEAR)
            color_depth = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
            cv2.imshow('depth',color_depth)

        if frame is not None:
            displayFrame("rgb", frame, depth_scaled)

        print("This round is %s seconds"%(time.time()-start_time))
        if cv2.waitKey(1) == ord('q'):
            GPIO.cleanup()
            break

