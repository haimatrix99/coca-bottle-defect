# Import library
import cv2
import numpy as np
from settings import *
from utils.img_to_np import img_to_np
from alibi_detect.utils.saving import load_detector

# Model path
model_path = './models/model_detector_vae'
#Load camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# Load model AE
od = load_detector(model_path)
# Classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

while True:
    # Detecting objects
    _ , frame = camera.read()
    height, width, _ = frame.shape
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(output_layers)
    
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

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'bottle':
                roi = frame[y:y+h, x:x+w]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = img_to_np(roi)
                roi = roi.astype('float32') / 255.0
                preds = od.predict(roi ,outlier_type='instance',    # use 'feature' or 'instance' level
                            return_feature_score=True,  # scores used to determine outliers
                            return_instance_score=True)
                if preds['data']['is_outlier'][0]:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), RED, 1)
                    cv2.putText(frame, 'Anomaly', (x-70,y+15), FONT, 0.5, RED, 1)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 1)
                    cv2.putText(frame, 'Normal', (x-70,y+15), FONT, 0.5, BLUE, 1)
    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()

