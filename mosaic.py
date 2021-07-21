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

mosaic_size = 2
path = "H:/i21k_coco/coco/"
json_filename = "yolov4_i21k_55_ann0.5.json"
output_path = "mosaic/"

max_width = 2000
max_height = 2000

with open(json_filename, "r") as read_file_json:
    loaded_json = json.load(read_file_json)

label_list = []
img_list = []

with open("instances_example.json", "r") as read_file_json:
    init_json = json.load(read_file_json)

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
        

final_image = np.zeros((max_height*mosaic_size, max_width*mosaic_size, 3),dtype=np.uint8)  
shift_x = 0
shift_y = 0
max_y = 0

x = 0
y = 0
id = 0

for image_id, image in images_dict.items():
    filename = image['file_name']
    filename = path + filename
    print(f"filename = {filename}")
    img = cv2.imread(filename)
    #print(img)

    m_id = id // (mosaic_size*mosaic_size)
    x = id % mosaic_size
    y = (id // mosaic_size) % mosaic_size

    if x==0 and y==0:
        shift_y += max_y
        if shift_x > 0 and shift_y > 0:
            result_image = final_image[0:shift_y, 0:+shift_x, :]
            #cv2.imshow('result_image', result_image)
            #cv2.waitKey(0)

            img_id_counter = m_id-1
            file_name = str(img_id_counter) + ".jpg"
            cv2.imwrite(output_path + file_name, result_image)

            img_list.append({
                "license": 1,
                "file_name": file_name,
                "coco_url": "",
                "height": result_image.shape[0],
                "width": result_image.shape[1],
                "date_captured": "2021-06-06 17:02:52",
                "flickr_url": "",
                "id": img_id_counter
            })
        shift_y = 0
        max_y = 0
        final_image = np.zeros((max_height*mosaic_size, max_width*mosaic_size, 3),dtype=np.uint8)
        final_image.fill(127)

    if x==0:
        shift_x = 0
        shift_y += max_y
        max_y = 0


    print(f" m_id = {m_id}, x = {x}, y = {y}, id = {id}, shift_x = {shift_x}, shift_y = {shift_y}, max_y = {max_y}")
    
    if image_id in ann_dict:
        for annotation in ann_dict[image_id]:
            category_id = annotation['category_id']
            color = (category_id*123 % 255, category_id*321 % 255, category_id*2 % 255)
    
            #bbox =  annotation['bbox']
            #left = int(bbox[0])
            #top =  int(bbox[1])
            #right = int(bbox[0]) + int(bbox[2])
            #bottom = int(bbox[1]) + int(bbox[3])

            annotation['image_id'] = m_id
            annotation['bbox'][0] += shift_x
            annotation['bbox'][1] += shift_y
            label_list.append(annotation)

    final_image[shift_y:img.shape[0]+shift_y, shift_x:img.shape[1]+shift_x, :] = img

    shift_x = shift_x + img.shape[1]
    max_y = max(max_y, img.shape[0])    
    id += 1

    #img = cv2.rectangle(img, (left, top), (right, bottom), color, 2)

    #cv2.imshow('image', img)
    #cv2.waitKey(0)

# save JSON
jimgs_annotations = {"images": img_list}
init_json.update(jimgs_annotations)

jlabels_annotations = {"annotations": label_list}
init_json.update(jlabels_annotations)

with open("out.json", 'w') as f:
    json.dump(init_json, f)

#print(init_json)

print("finish")