# -*- coding: utf-8 -* -
"""
pascal voc转coco格式
"""
import os
import shutil
from glob import glob
import json
import datetime as d
import mmcv
from lxml import etree
from utils import dataset_util
from utils import label_map_util

def _reset_dir(dist_dir):
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    shutil.rmtree(dist_dir)

    dist_anno_path = os.path.join(dist_dir, "annotations")
    dist_image_train_path = os.path.join(dist_dir, "images","train")
    dist_image_val_path = os.path.join(dist_dir, "images","val")
    dist_image_test_path = os.path.join(dist_dir, "images","test")

    os.makedirs(dist_anno_path)
    os.makedirs(dist_image_train_path)
    os.makedirs(dist_image_val_path)
    os.makedirs(dist_image_test_path)

    return dist_anno_path,dist_image_train_path,dist_image_val_path,dist_image_test_path

def _load_label_map(label_map_path):
    '''
    加载label，生成 {类id : 类名} 的映射字典
    :return:
    '''
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
    category_name_index = {category_index[cat_id]["name"]: category_index[cat_id] for cat_id in category_index}
    categories = label_map_util.create_categories_from_labelmap(label_map_path)

    return category_index,category_name_index,categories

def _load_anno_sample(anno_path):
    '''
    加载一个标注信息
    :param anno_path: pascal voc 格式标注文件路径
    :return:
    '''
    with open(anno_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    anno_data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    return anno_data

def pascal_to_coco(data_dir,dist_dir,label_map_path):
    dist_anno_path, dist_image_train_path, dist_image_val_path, dist_image_test_path = _reset_dir(dist_dir)
    category_index, category_name_index, categories = _load_label_map(label_map_path)

    train_samples = mmcv.list_from_file(os.path.join(data_dir,"ImageSets","Main","train.txt"))
    val_samples = mmcv.list_from_file(os.path.join(data_dir,"ImageSets","Main","val.txt"))

    def cc(cat):
        cat["supercategory"] = cat["name"]
        return cat

    categories = list(map(cc,categories))

    instance_val = {"images":[],"type": "instances","annotations":[],"categories":categories}
    instance_train = {"images":[],"type": "instances","annotations":[],"categories":categories}

    image_id = 0
    anno_id = 0

    for id,sample in enumerate(val_samples + train_samples):
        image_file = os.path.join(data_dir,"JPEGImages",sample + ".jpg")
        anno_file = os.path.join(data_dir,"Annotations",sample + ".xml")

        anno_data = _load_anno_sample(anno_file)

        image_id += 1

        image_instance = {
            "license": 1,
            "url": "",
            "file_name": os.path.basename(image_file),
            "height": anno_data["size"]["width"],
            "width": anno_data["size"]["height"],
            "date_captured": d.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id":image_id,
        }

        anno_list = []
        if "object" in anno_data:
            for object in anno_data["object"]:
                anno_id += 1
                anno_list.append({
                    "segmentation":[],
                    "area":0,
                    "iscrowd":0,
                    "image_id":image_id,
                    "category_id":category_name_index[object["name"]]["id"],
                    "id": anno_id,
                    # [x,y,width,height]
                    "bbox":[float(object["bndbox"]["xmin"]),float(object["bndbox"]["ymin"]),float(int(object["bndbox"]["xmax"])-int(object["bndbox"]["xmin"])),float(int(object["bndbox"]["ymax"])-int(object["bndbox"]["ymin"]))],
                })

        if sample in val_samples:
            shutil.copyfile(image_file,os.path.join(dist_image_val_path,os.path.basename(image_file)))
            shutil.copyfile(image_file,os.path.join(dist_image_test_path,os.path.basename(image_file)))
            instance_val["images"].append(image_instance)
            instance_val["annotations"] += anno_list
        if sample in train_samples:
            shutil.copyfile(image_file,os.path.join(dist_image_train_path,os.path.basename(image_file)))
            instance_train["images"].append(image_instance)
            instance_train["annotations"] += anno_list

    instance_test = instance_val.copy()

    json.dump(instance_train,open(os.path.join(dist_anno_path,"instances_train.json"),"w"))
    json.dump(instance_val,open(os.path.join(dist_anno_path,"instances_val.json"),"w"))
    json.dump(instance_test,open(os.path.join(dist_anno_path,"instances_test.json"),"w"))

if __name__ == '__main__':
    pascal_to_coco(
        data_dir="/Users/rensike/Work/友极/cexie",
        dist_dir="/Users/rensike/Work/友极/coco_cexie",
        label_map_path="/Users/rensike/Work/友极/cexie/label_map.pbtxt"
    )

