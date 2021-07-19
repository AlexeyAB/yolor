import argparse
import glob
import json
import os
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm
import collections

import cv2

path = "H:/i21k_coco/coco/"
json_filename = "yolov4_i21k_55_ann0.5.json"

with open(json_filename, "r") as read_file_json:
    loaded_json = json.load(read_file_json)


images_dict = {}
# make dict images: image_id - value
for image in loaded_json['images']:
    image_id = image['id']
    images_dict[image_id] = image


ann_dict = {}
# make dict of dicts: category - score - value
for annotation in loaded_json['annotations']:
    image_id = annotation['image_id']
    if image_id in ann_dict:
        ann_dict[image_id].append(annotation)
    else:
        ann_dict[image_id] = [annotation]
        


for image_id, image in images_dict.items():
    filename = image['file_name']
    filename = path + filename
    print(f"filename = {filename}")
    img = cv2.imread(filename)
    #print(img)

    if image_id in ann_dict:
        for annotation in ann_dict[image_id]:
            category_id = annotation['category_id']
            color = (category_id*123 % 255, category_id*321 % 255, category_id*2 % 255)
            
            bbox =  annotation['bbox']
            left = int(bbox[0])
            top =  int(bbox[1])
            right = int(bbox[0]) + int(bbox[2])
            bottom = int(bbox[1]) + int(bbox[3])

            img = cv2.rectangle(img, (left, top), (right, bottom), color, 2)

    cv2.imshow('image', img)
    cv2.waitKey(0)