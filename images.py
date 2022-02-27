import cv2
import sys
import os
from utils.webcam import Webcam
from utils.increment_path import increment_path
from argparse import ArgumentParser
from pathlib import Path



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

parser = ArgumentParser()
parser.add_argument("--webcam", default=0,type=int, help='Source webcam 0/1/2')
parser.add_argument("--num-images", default=100, type=int, help="Number images to save the file")
parser.add_argument("--show", default=False, action='store_true', help="Show webcam")
parser.add_argument("--project", default=ROOT / 'images', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument("--get", default=False, action='store_true', help="ON/OFF get images")


args = vars(parser.parse_args())

cap = Webcam(stream_id=args['webcam'])
cap.start()
if args["get"]:
    save_dir = increment_path(Path(args['project']) / args['name'] , exist_ok=False, mkdir=True)  # increment run

index = 0

while True:
    if cap.stopped is True:
        break
    else:
        image = cap.read()
    if args['get']:
        cv2.imwrite(f'{save_dir}/{save_dir.stem}-{index:03d}.jpg', image)
        index+=1
    if args['show']:
        cv2.imshow("Webcam", image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if index == args["num_images"]:
        print('[INFO] Done')
        break

cap.stop()
cv2.destroyAllWindows()
    
