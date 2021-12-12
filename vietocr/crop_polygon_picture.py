import numpy as np
import cv2
import glob
import os
from os import walk
import argparse

parser = argparse.ArgumentParser()
# PATH
parser.add_argument("--detected_bboxes", default="/content/detect/")
parser.add_argument("--original_images", default="/content/TestA/")
parser.add_argument("--output_cropped", default="/content/crop_polygon_picture/")

# input args need three items:
# detected bboxes
# original images
# output cropped image

args, unknown = parser.parse_known_args()

detected_path = args.detected_bboxes
original_path = args.original_images
output_path   = args.output_cropped

# CROP POLYGON PICTURE
#os.mkdir(output_path)

# Crop polygon picture
os.chdir(output_path)

filenames = []
filenames_txt = next(walk(detected_path), (None, None, []))[2]

for item in filenames_txt:
    temp = item.replace(".txt", "")
    filenames.append(temp)

for i in range(len(filenames_txt)):
    url = original_path + filenames[i]
    path_file = detected_path + filenames[i] + ".txt"
    f = open(path_file, "r")
    count = 1
    image = cv2.imread(url)
    for item in f:
        temp_lst = item.split(',')
        temp = filenames[i].split('.')
        if temp[1]=="jpg":
            filename =  temp[0] + "-" + str(count) + ".jpg"
        else:
            filename = temp[0] + "-" + str(count) + ".jpeg"
        #print(filename)
        temp_Item = [int(temp_lst[i]) for i in range(8)]
        temp_Arr = np.array(temp_Item)
        res = np.reshape(temp_Arr, (-1, 2))

        pts = res

        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        croped = image[y:y + h, x:x + w].copy()

        ## (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)

        ## (4) add the white background
        bg = np.ones_like(croped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst2 = bg + dst

        if croped.size !=0 :
            cv2.imwrite(filename, dst2)
            count = count + 1
    f.close()