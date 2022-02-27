import cv2
import os
import time
import sys
import glob
import numpy as np
import argparse
from pathlib import Path
from imutils.settings import *
from imutils.img_to_np import *
from imutils.increment_path import *
from imutils.webcam import Webcam
from alibi_detect.utils.saving import load_detector
from alibi_detect.utils.visualize import plot_feature_outlier_image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

index = 0

def pred_obj(results):
    if len(results) == args["num_frames"]:
        if np.mean(results) > od.threshold:
            print("[INFO] Object detect is Anomaly")
            results.clear()
            return 1
        else:
            print("[INF] Object detect is Normaly")
            results.clear()  
            return 0
    else:
        return None
        
def detect_image(image):
    result, roi, preds = None, None, None
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
            if confidence > args["conf_thres"] and label == 'bottle':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2 if center_x >= w/2 else w/2 - center_x)
                y = int(center_y - h / 2 if center_y >= h/2 else h/2 - center_y)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if args['save_crop']:
                global index
                cv2.imwrite(f'{save_dir}/{save_dir.stem}-{index:04d}.jpg', image[y:y+h, x:x+w])
                print(f"[INFO] {save_dir}/{save_dir.stem}-{index:04d}.jpg saved")
                index += 1
            roi = image[y:y+h, x:x+w]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = img_to_np(roi)
            roi = roi.astype('float32') / 255.0
            preds = od.predict(roi ,outlier_type='feature',
                        return_feature_score=True,
                        return_instance_score=True)
            result = preds['data']['instance_score'][0]
    return image, result, roi, preds

def run():
    results = []
    if args['source'].isnumeric():
        cap = Webcam(src=int(args['source'])) # 0 id for main camera
        cap.start()
        print("[INFO] Camera Opened")
        start = time.time()
        while True:
            if cap.stopped is True:
                break
            else:
                image = cap.read()
            image, result, roi, preds = detect_image(image)
            
            if result != None:
                results.append(result)
                
            res = pred_obj(results)
            
            if res != None:
                if args["show_plot"]:
                    x_recon = od.vae(roi).numpy()
                    plot_feature_outlier_image(preds, roi, x_recon, max_instances=1, n_channels=1)
                cv2.putText(image, "Anomaly" if res else "Normaly", (0,30), FONT, 1, RED if res else BLUE, 2)

            if args['show_image']:
                cv2.imshow("Webcam", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        end = time.time() - start
        print(f"[INFO] Time elapsed: {end//60} mins {end%60}")
        cap.stop()
        cv2.destroyAllWindows()
        
    else:
        print("[INFO] Starting detect image")
        start = time.time()
        p = str(Path(args['source']).resolve())
        if '*' in p:
            filenames = glob.glob(p, recursive=True)
        elif os.path.isdir(p):
            filenames = glob.glob(os.path.join(p, '*.jpg'))
        elif os.path.isfile(p):
            filenames = [p]
        for filename in filenames:
            image = cv2.imread(filename)
            image, result, roi, preds = detect_image(image)
            if result != None:    
                while len(results) < args["num_frames"]: 
                    results.append(result)
                   
            res = pred_obj(results)
            
            if res != None:
                if args["show_plot"]:
                    x_recon = od.vae(roi).numpy()
                    plot_feature_outlier_image(preds, roi, x_recon, max_instances=1, n_channels=1)
                cv2.putText(image, "Anomaly-" + str(preds["data"]["instance_score"][0]) if res else "Normaly-"+ str(preds["data"]["instance_score"][0]), (0,30), FONT, 1, RED if res else BLUE, 2)

            if args['show_image']:
                cv2.imshow("Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        end = time.time() - start
        print(f"[INFO] Time elapsed: {int(end//60)} mins {int(end%60)} secs")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default='0', help="Source 0,1,2/folder/file")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Conf threshold")
    parser.add_argument("--detect-thres", type=float, default=0.002776, help="Anomaly threshold")
    parser.add_argument("--num-frames", default=5,type=int, help="Number frames want to detect object")
    parser.add_argument("--model", default="models/model_detector_vae", help="File model detect")
    parser.add_argument("--weights", default="models/yolov3-spp.weights", help="File weight")
    parser.add_argument("--conf", default="models/yolov3.cfg", help="File config")
    parser.add_argument("--names", default="models/coco.names", help="File classes")
    parser.add_argument("--project", default= ROOT / 'crops', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument("--save-crop", default=False,action='store_true', help="Save crop object in images")
    parser.add_argument("--show-image", default=False, action='store_true', help="Show image result")
    parser.add_argument("--show-plot", default=False,action='store_true', help="Save label in images")

    args = vars(parser.parse_args())
    
    if args['save_crop']:
        save_dir = increment_path(Path(args['project']) / args['name'] , exist_ok=False, mkdir=True)  # increment run
        
    classes = []
    with open(args['names'], "r") as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet(args["weights"],args["conf"])
    
    od = load_detector(args["model"])
    od.threshold = args["detect_thres"]
    
    img = cv2.imread("images/exp/exp-099.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_np(img)
    img = img.astype('float32') / 255.0
    
    preds = od.predict(img ,outlier_type='feature',
                return_feature_score=False,
                return_instance_score=False)
    
    print("[INFO] Initialize Sucessefully")  
    print(f"[INFO] Outlier detection threshold: {od.threshold}")
    
    run()