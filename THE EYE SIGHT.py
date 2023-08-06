import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
import pyttsx3
ran=0
engine = pyttsx3.init()
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
TRIG = 4
ECHO = 17
vibration=14
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)
GPIO.setup(vibration,GPIO.OUT)
while True:
    GPIO.output(TRIG, False)
    time.sleep(0)
    GPIO.output(TRIG, True)
    time.sleep(0)
    GPIO.output(TRIG, False)
    while GPIO.input(ECHO)==0:
        pulse_start=time.time()

    while GPIO.input(ECHO)==1:
        pulse_end=time.time()

    pulse_duration=pulse_end-pulse_start

    distance = pulse_duration*11150
    distance = round(distance,2)
    if distance < 50:
        GPIO.output(vibration,GPIO.HIGH)
        print("VIBRATION ON")
        video_capture = cv2.VideoCapture(0)
        while True:
        # Capture frame-by-frame
            re,img = video_capture.read()
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            while ran==0:
                
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                        # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                        # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                print(indexes)
                font = cv2.FONT_HERSHEY_PLAIN
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        color = colors[class_ids[i]]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                        print(label,"detected")
                        engine.say(label)
                        engine.say("detected at")
                        engine.say(distance)
                        engine.say("centimetre")
                        engine.runAndWait()
                        ran=ran+1



            cv2.imshow("Image",cv2.resize(img,(800,600)))
            if cv2.waitKey(1)& 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

       
    else:
       
        print("VIBRATION OFF")
        GPIO.output(vibration,GPIO.LOW)