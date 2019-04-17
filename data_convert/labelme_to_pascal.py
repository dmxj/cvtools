# -*- coding: utf-8 -* -
"""
labelme 数据格式转pascal voc
"""

import os
import shutil
import glob
from xml.etree import ElementTree as etree
from xml.etree.ElementTree import Element, SubElement, ElementTree
from utils import dataset_util
import cv2
import json
import random
import numpy as np

def reset_dir(dist_dir):
    """
    pascal voc 目录
    :param dist_dir:
    :return:
    """
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    shutil.rmtree(dist_dir)

    dist_image_path = os.path.join(dist_dir, "JPEGImages")
    dist_anno_path = os.path.join(dist_dir, "Annotations")
    dist_imageset_path = os.path.join(dist_dir, "ImageSets", "Segmentation")
    dist_segclass_path = os.path.join(dist_dir, "SegmentationClass")
    dist_segobj_path = os.path.join(dist_dir, "SegmentationObject")

    os.makedirs(dist_image_path)
    os.makedirs(dist_anno_path)
    os.makedirs(dist_imageset_path)
    os.makedirs(dist_segclass_path)
    os.makedirs(dist_segobj_path)

    return dist_image_path, dist_anno_path, dist_imageset_path, dist_segclass_path,dist_segobj_path

def map_to_xml(anno_data, dist_path):
    """
    写标注
    :param anno_data:
    :param dist_path:
    :return:
    """
    assert "folder" in anno_data
    assert "filename" in anno_data
    assert "size" in anno_data
    assert "width" in anno_data["size"]
    assert "height" in anno_data["size"]
    assert "depth" in anno_data["size"]

    segmented = "1"
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
    tree.write(dist_path, encoding='utf-8')

def get_bbox_from_points(points):
    """
    根据polygon顶点获取bbox
    :param points:
    :return:
    """
    x_list = [p[0] for p in points]
    y_list = [p[1] for p in points]

    xmin = min(x_list)
    ymin = min(y_list)
    xmax = max(x_list)
    ymax = max(y_list)

    return {
        "xmin":xmin,
        "ymin":ymin,
        "xmax":xmax,
        "ymax":ymax,
    }


def draw_mask_png(source_dir):
    """
    将labelme标注得到的polygon生成对应的mask图片
    :param source_dir:
    :return:
    """

    from PIL import Image,ImageDraw

    anno_files = glob.glob(os.path.join(source_dir, "*.json"))

    segment_path = os.path.join(source_dir,"segment")
    if os.path.exists(segment_path):
        shutil.rmtree(segment_path)
    os.makedirs(segment_path)

    for anno_file in anno_files:
        sample_name = os.path.basename(anno_file).rsplit(".", 1)[0]

        image_path = os.path.join(source_dir, "{}.png".format(sample_name))

        im_raw = cv2.imread(image_path)
        height, width, dim = im_raw.shape

        anno_data = json.load(open(anno_file,"r"))
        shapes = anno_data["shapes"]

        mask_png_path = os.path.join(segment_path,"{}.png".format(sample_name))

        img = np.zeros([height, width,3], np.uint8)

        im = Image.fromarray(img.astype('uint8')).convert('RGB')

        draw = ImageDraw.Draw(im)

        for shape in shapes:
            points = shape["points"]
            points = [tuple(p) for p in points]
            draw.polygon(points,fill="blue")
        im.save(mask_png_path)

    print("write png done.")


def make_pascal_voc(source_dir, dist_dir):
    dist_image_path, dist_anno_path, dist_imageset_path, dist_segclass_path, dist_segobj_path = reset_dir(dist_dir)
    anno_files = glob.glob(os.path.join(source_dir,"*.json"))

    trainval_list = []

    for anno_file in anno_files:
        sample_name = os.path.basename(anno_file).rsplit(".", 1)[0]
        image_path = os.path.join(source_dir,"{}.png".format(sample_name))

        im_raw = cv2.imread(image_path)
        height, width, dim = im_raw.shape
        anno_dict = json.load(open(anno_file,"r"))

        anno_data = {
            "folder": dist_image_path,
            "filename": os.path.basename(image_path),
            "size": {
                "width": width,
                "height": height,
                "depth": dim
            },
            "objects": []
        }

        shapes = anno_dict["shapes"]
        for shape in shapes:
            anno_data["objects"].append({
                "name": shape["label"],
                "bndbox": get_bbox_from_points(shape["points"])
            })

        shutil.copyfile(image_path, os.path.join(dist_image_path, os.path.basename(image_path)))

        seg_image_name = os.path.basename(image_path)
        shutil.copyfile(os.path.join(source_dir,"segment",seg_image_name),os.path.join(dist_segclass_path, seg_image_name))
        shutil.copyfile(os.path.join(source_dir,"segment",seg_image_name),os.path.join(dist_segobj_path, seg_image_name))

        dist_anno_file = os.path.join(dist_anno_path, "{}.xml".format(sample_name))
        map_to_xml(anno_data, dist_anno_file)
        trainval_list.append(sample_name)

    print("write done, total num:{}".format(len(trainval_list)))

    train_list = trainval_list
    val_list = trainval_list

    with open(os.path.join(dist_imageset_path, "trainval.txt"), "w+") as fw:
        fw.write("\n".join(trainval_list) + "\n")

    with open(os.path.join(dist_imageset_path, "train.txt"), "w+") as fw:
        fw.write("\n".join(train_list) + "\n")

    with open(os.path.join(dist_imageset_path, "val.txt"), "w+") as fw:
        fw.write("\n".join(val_list) + "\n")

    print("write done")

if __name__ == '__main__':
    make_pascal_voc(
        source_dir="",
        dist_dir="",
    )







