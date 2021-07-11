import argparse
import glob
import json
import os
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm
import collections

# Keep only the first N labels in JSON file with the Highest/Lowest confidence-score
# set params
num_elements = 2000
top_flag = True     # get Top or Bottom N elements
json_filename = "yolov4_cocounlabeled_55_ann0.5.json"


with open(json_filename, "r") as read_file_json:
    loaded_json = json.load(read_file_json)

with open("instances_example.json", "r") as read_file_json:
    init_json = json.load(read_file_json)



"""
{"segmentation": {}, 
"area": 327968, 
"iscrowd": 0, 
"image_id": 92754, 
"bbox": [456.13, 0.6, 94.67, 116.54], 
"category_id": 44, 
"id": 514025, 
"score": 0.55518},
"""

"""
"images": 
[
    {
        "license": 1, 
        "file_name": "000000441778.jpg", 
        "coco_url": "", 
        "height": 69, 
        "width": 640, 
        "date_captured": "2021-06-06 17:02:52", 
        "flickr_url": "", 
        "id": 1
    }
    ,,,
]
 """


ann_dict = {}

# add categories Id
for category in init_json['categories']:
    category_id = category['id']
    ann_dict[category_id] = {}


# make dict of dicts: category - score - value
for annotation in loaded_json['annotations']:
    category_id = annotation['category_id']
    score = annotation['score']
    ann_dict[category_id][score] = annotation


# sort by scores
for category_key, dict_value in ann_dict.items():
    sorted_first_N_vals = [dict_value[k] for k in sorted(dict_value.keys(), reverse=top_flag)[:num_elements]]

    ann_dict[category_key] = sorted_first_N_vals
    #print(f"category = {category}")


# concate lists
label_list = []
for category_key, dict_value in ann_dict.items():
    label_list += dict_value
    #print(f"{dict_value} \n")


images_dict = {}
# make dict images: image_id - value
for image in loaded_json['images']:
    image_id = image['id']
    images_dict[image_id] = image


# list of IDs of images
img_ids = []
for label in label_list:
    img_ids.append(label['image_id'])

# keep unique IDs
img_ids = list(set(img_ids))

# list of images
img_list = []
img_list = [images_dict[k] for k in img_ids]


# save JSON
jimgs_annotations = {"images": img_list}
init_json.update(jimgs_annotations)

jlabels_annotations = {"annotations": label_list}
init_json.update(jlabels_annotations)

with open("out.json", 'w') as f:
    json.dump(init_json, f)

#print(init_json)

print("finish")


