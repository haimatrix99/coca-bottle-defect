import cv2
import numpy as np
import argparse
from imutils.img_to_np import *
from imutils.settings import *
from alibi_detect.utils.saving import load_detector


parser = argparse.ArgumentParser()
parser.add_argument("--conf-thresh", type=float, default=0.5, help="Conf threshold")
parser.add_argument("--iou-thresh", type=float, default=0.5, help="Conf threshold")
parser.add_argument("--detect-thresh", type=float, default=0.002776, help="Anomaly threshold")
parser.add_argument("--model", default="models/model_detector_vae", help="File model detect")
parser.add_argument("--weights", default="models/yolov3-spp.weights", help="File weight")
parser.add_argument("--conf", default="models/yolov3.cfg", help="File config")
parser.add_argument("--names", default="models/coco.names", help="File classes")

args = vars(parser.parse_args())
   
classes = []
with open(args['names'], "r") as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(args["weights"],args["conf"])

od = load_detector(args["model"])
od.threshold = args["detect_thresh"]

img = cv2.imread("data/exp/exp-099.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img_to_np(img)
img = img.astype('float32') / 255.0

preds = od.predict(img ,outlier_type='feature',
            return_feature_score=False,
            return_instance_score=False)

print("[INFO] Initialize Sucessefully")  
print(f"[INFO] Outlier detection threshold: {od.threshold}")

def detect_image(image):
    result = None
    height, width = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1/255, (608, 608), (0, 0, 0), True, crop=False)
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
            label = str(classes[class_id])
            if confidence > args["conf_thresh"] and label == 'bottle':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2 if center_x >= w/2 else w/2 - center_x)
                y = int(center_y - h / 2 if center_y >= h/2 else h/2 - center_y)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, args["iou_thresh"], 0.4)
    
    for i in range(len(boxes)):
        if i in indexes:
            roi = image[y:y+h, x:x+w]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = img_to_np(roi)
            roi = roi.astype('float32') / 255.0
            preds = od.predict(roi ,outlier_type='feature',
                        return_feature_score=True,
                        return_instance_score=True)
            result = preds['data']['instance_score'][0]
            cv2.rectangle(image, (x,y), (x+w,y+h), RED if result > od.threshold else BLUE, 2)
    return image, result

        

