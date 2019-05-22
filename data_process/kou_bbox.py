#!/usr/bin/env python
# coding=utf-8
import cv2
import glob
import os
from lxml import etree
from utils import dataset_util
from tqdm import tqdm
import shutil

def _load_anno_sample(anno_path):
    with open(anno_path, 'r') as fid:
        xml_str = fid.read()
    xml_str = etree.fromstring(xml_str)
    anno_data = dataset_util.recursive_parse_xml_to_dict(xml_str)['annotation']
    return anno_data

def koutu(source_dir, out_dir):
    image_files = glob.glob(os.path.join(source_dir, "JPEGImages", "*.jpg"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    total_hongtiepi = 0
    total_h = 0
    total_w = 0
    for image_path in tqdm(image_files):
         sample_name = os.path.basename(image_path).rsplit(".", 1)[0]
         anno_path = os.path.join(source_dir, "Annotations", "{}.xml".format(sample_name))
         anno_data = _load_anno_sample(anno_path)
         image = cv2.imread(image_path)
         if "object" in anno_data:
             flag = 1
             for obj in anno_data['object']:
                 if obj['name'] == "hongtiepi":
                    ymin = int(obj['bndbox']['ymin'])
                    xmin = int(obj['bndbox']['xmin'])
                    ymax = int(obj['bndbox']['ymax'])
                    xmax = int(obj['bndbox']['xmax'])
                    total_hongtiepi +=1
                    total_h += ymax-ymin
                    total_w += xmax-xmin
                    flag +=1
                    bengque_img = image[ymin:ymax, xmin:xmax]
                    image_name = str(flag) + "_" + os.path.basename(image_path)
                    save_path = os.path.join(out_dir, image_name)
                    cv2.imwrite(save_path, bengque_img)
    print("total_hongtiepi:", total_hongtiepi)
    print("mean_h:", int(total_h/total_hongtiepi))
    print("mean_w:", int(total_w/total_hongtiepi))

if __name__ == "__main__":
    source_dir ="/Users/rensike/Work/宝钢热轧/baogang_hot_data_v4/pascal_voc_hot_v3_3"
    out_dir = "/Users/rensike/Work/宝钢热轧/baogang_hot_data_v4/pascal_voc_hot_v3_3_bbox/hongtiepi"
    koutu(source_dir, out_dir)
