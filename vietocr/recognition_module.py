import numpy as np
import cv2
import glob
import os
from os import walk
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from natsort import natsorted
parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="/content/weights/transformerocr.pth")
parser.add_argument("--drop_score", default=0.8)
# PATH
parser.add_argument("--cropped_bboxes", default="/content/crop_polygon_picture/")
parser.add_argument("--seperated", default="/content/separate/")
parser.add_argument("--seperated_null", default="/content/separate_null/")
parser.add_argument("--original_detect", default="/content/separate_null/")
parser.add_argument("--original_images", default="/content/separate_null/")

args, unknown = parser.parse_known_args()

config = Cfg.load_config_from_name('vgg_transformer')

config['weights'] = args.weights
config['cnn']['pretrained'] = False
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False
config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° '

detector = Predictor(config)

detect_files = glob.glob(args.original_detect + "*")

for detect_file in detect_files:
  detect_filename = os.path.basename(detect_file)
  original_name = detect_filename.split(".")[0]
  bboxes_path = natsorted(glob.glob(args.cropped_bboxes + original_name + "-*"))
  bboxes_labels = []

  for bbox_path in bboxes_path:
    img = Image.open(bbox_path)
    text, prob = detector.predict(img, return_prob=True)
    text = ("###" if prob < float(args.drop_score) else text)
    bboxes_labels.append(text)
  
  # read from original detect
  det_f = open(detect_file, 'r', encoding="utf-8")
  lines = det_f.read().split("\n")
  det_f.close()

  seperated_content = ""
  seperated_null_content = ""

  # loop parralel
  for i in range(min(len(lines), len(bboxes_labels))):
    if (lines[i] == ""):
      break
    boxes = lines[i].split(",")
    while (len(boxes) < 9):
      boxes.append("")
    boxes[8] = bboxes_labels[i]
    if (boxes[8] == "###"):
      for i, box in enumerate(boxes):
        seperated_null_content += box
        seperated_null_content += ('\n' if i == 8 else ',')
    else:
      for i, box in enumerate(boxes):
        seperated_content += box
        seperated_content += ('\n' if i == 8 else ',')

  # write seperated
  f = open(args.seperated + detect_filename, 'w', encoding="utf-8")
  f.write(seperated_content)
  f.close()

  # write seperated null
  f = open(args.seperated_null + detect_filename, 'w', encoding="utf-8")
  f.write(seperated_null_content)
  f.close()

  print("Done", detect_filename)