# -*- coding: utf-8 -* -
"""
coco格式数据转Pascal
"""
import os
import json
import shutil
from lxml import etree
from lxml.etree import Element,SubElement,ElementTree

def reset_dir(dist_dir):
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    shutil.rmtree(dist_dir)

    dist_image_path = os.path.join(dist_dir, "JPEGImages")
    dist_anno_path = os.path.join(dist_dir, "Annotations")
    dist_imageset_path = os.path.join(dist_dir, "ImageSets", "Main")

    os.makedirs(dist_image_path)
    os.makedirs(dist_anno_path)
    os.makedirs(dist_imageset_path)

    return dist_image_path, dist_anno_path, dist_imageset_path

def map_to_xml(anno_data, dist_path):
    assert "folder" in anno_data
    assert "filename" in anno_data
    assert "size" in anno_data
    assert "width" in anno_data["size"]
    assert "height" in anno_data["size"]
    assert "depth" in anno_data["size"]
    assert "objects" in anno_data

    segmented = "0"
    pose = "Unspecified"
    truncated = "0"
    difficult = "0"

    root = Element("annotation")
    SubElement(root, 'folder').text = anno_data["folder"]
    SubElement(root, 'filename').text = anno_data["filename"]

    source = SubElement(root, 'source')
    SubElement(source, 'database').text = "The VOC2007 Database"
    SubElement(source, 'annotation').text = "PASCAL VOC2007"
    SubElement(source, 'image').text = "flickr"

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(anno_data["size"]["width"])
    SubElement(size, 'height').text = str(anno_data["size"]["height"])
    SubElement(size, 'depth').text = str(anno_data["size"]["depth"])

    SubElement(root, 'segmented').text = segmented

    for object in anno_data["objects"]:
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = object["name"]
        SubElement(obj, 'pose').text = pose
        SubElement(obj, 'truncated').text = truncated
        SubElement(obj, 'difficult').text = difficult
        bndbox = SubElement(obj, 'bndbox')
        SubElement(bndbox, 'xmin').text = str(object["bndbox"]["xmin"])
        SubElement(bndbox, 'ymin').text = str(object["bndbox"]["ymin"])
        SubElement(bndbox, 'xmax').text = str(object["bndbox"]["xmax"])
        SubElement(bndbox, 'ymax').text = str(object["bndbox"]["ymax"])

    tree = ElementTree(root)
    tree.write(dist_path, encoding='utf-8',pretty_print=True)

def coco_to_voc(image_dir,anno_path,split="train"):
    anno_data = json.load(open(anno_path,"r"))
    images = anno_data["images"]
    annos = anno_data["annotations"]
    anno_dict = {}

    for anno in annos:
        if anno["image_id"] not in anno_dict:
            anno_dict[anno["image_id"]] = []
        anno_dict[anno["image_id"]].append(anno)

    for image_file in images:
        pass

