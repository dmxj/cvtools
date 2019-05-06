# -*- coding: utf-8 -* -
"""
labelme 数据格式转pascal voc
"""

import os
import shutil
from lxml import etree
from lxml.etree import Element, SubElement, ElementTree
from utils import dataset_util
import cv2
import json
import random
import numpy as np


def reset_dir(dist_dir, include_seg=False):
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
    if include_seg:
        dist_imageset_path = os.path.join(dist_dir, "ImageSets", "Segmentation")
    else:
        dist_imageset_path = os.path.join(dist_dir, "ImageSets", "Main")
    dist_segclass_path = os.path.join(dist_dir, "SegmentationClass")
    dist_segobj_path = os.path.join(dist_dir, "SegmentationObject")

    os.makedirs(dist_image_path)
    os.makedirs(dist_anno_path)
    os.makedirs(dist_imageset_path)
    if include_seg:
        os.makedirs(dist_segclass_path)
        os.makedirs(dist_segobj_path)

    return dist_image_path, dist_anno_path, dist_imageset_path, dist_segclass_path, dist_segobj_path


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
    assert "objects" in anno_data

    segmented = anno_data["segmented"]
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
    tree.write(dist_path, encoding='utf-8', pretty_print=True)


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
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def draw_mask_png(source_dir):
    """
    将labelme标注得到的polygon生成对应的mask图片
    :param source_dir:
    :return:
    """

    from PIL import Image, ImageDraw

    anno_files = glob.glob(os.path.join(source_dir, "*.json"))

    segment_path = os.path.join(source_dir, "segment")
    if os.path.exists(segment_path):
        shutil.rmtree(segment_path)
    os.makedirs(segment_path)

    for anno_file in anno_files:
        sample_name = os.path.basename(anno_file).rsplit(".", 1)[0]

        image_path = os.path.join(source_dir, "{}.png".format(sample_name))

        im_raw = cv2.imread(image_path)
        height, width, dim = im_raw.shape

        anno_data = json.load(open(anno_file, "r"))
        shapes = anno_data["shapes"]

        mask_png_path = os.path.join(segment_path, "{}.png".format(sample_name))

        img = np.zeros([height, width, 3], np.uint8)

        im = Image.fromarray(img.astype('uint8')).convert('RGB')

        draw = ImageDraw.Draw(im)

        for shape in shapes:
            points = shape["points"]
            points = [tuple(p) for p in points]
            draw.polygon(points, fill="blue")
        im.save(mask_png_path)

    print("write png done.")


def make_pascal_voc(
        image_anno_path,
        dist_dir,
        include_seg=False,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
):
    """
    labelme 格式数据转pascal voc
    :param source_dir: [(image_path_0,anno_path_0),(image_path_1,anno_path_1),......]
    :param dist_dir:
    :param include_seg:
    :param train_ratio:
    :param val_ratio:
    :param test_ratio:
    :return:
    """
    if include_seg:
        assert len(image_anno_path[0]) == 3, "you should include seg image path!"
    assert (train_ratio + val_ratio + test_ratio) == 1.0, "invalid ratio params!"

    dist_image_path, dist_anno_path, dist_imageset_path, dist_segclass_path, dist_segobj_path = reset_dir(dist_dir,
                                                                                                          include_seg)

    trainval_list = []

    for path_item in image_anno_path:
        if include_seg:
            (image_file, anno_file, seg_image_file) = path_item
        else:
            (image_file, anno_file) = path_item
        sample_name = os.path.basename(anno_file).rsplit(".", 1)[0]

        im_raw = cv2.imread(image_file)
        height, width, dim = im_raw.shape
        anno_dict = json.load(open(anno_file, "r"))

        anno_data = {
            "segmented": "1" if include_seg else "0",
            "folder": dist_image_path,
            "filename": os.path.basename(image_file),
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

        if len(anno_data["objects"]) == 0:
            print("{} no have bbox".format(image_file))

        shutil.copyfile(image_file, os.path.join(dist_image_path, os.path.basename(image_file)))

        if include_seg:
            seg_image_name = os.path.basename(image_file)
            shutil.copyfile(seg_image_file, os.path.join(dist_segclass_path, seg_image_name))
            shutil.copyfile(seg_image_file, os.path.join(dist_segobj_path, seg_image_name))

        dist_anno_file = os.path.join(dist_anno_path, "{}.xml".format(sample_name))
        map_to_xml(anno_data, dist_anno_file)
        trainval_list.append(sample_name)

    print("write done, total num:{}".format(len(trainval_list)))

    # TODO:
    train_list = random.sample(trainval_list, int(len(trainval_list) * train_ratio))
    if test_ratio == 0:
        val_list = [s for s in trainval_list if s not in train_list]
        test_list = val_list
    else:
        val_list = random.sample([s for s in trainval_list if s not in train_list], int(len(trainval_list) * val_ratio))
        test_list = [s for s in trainval_list if s not in train_list and s not in val_list]

    with open(os.path.join(dist_imageset_path, "trainval.txt"), "w+") as fw:
        fw.write("\n".join(trainval_list) + "\n")

    with open(os.path.join(dist_imageset_path, "train.txt"), "w+") as fw:
        fw.write("\n".join(train_list) + "\n")

    with open(os.path.join(dist_imageset_path, "val.txt"), "w+") as fw:
        fw.write("\n".join(val_list) + "\n")

    with open(os.path.join(dist_imageset_path, "test.txt"), "w+") as fw:
        fw.write("\n".join(test_list) + "\n")

    print("write done")


if __name__ == '__main__':
    from glob import glob

    anno_files = glob(os.path.join("/Users/rensike/Work/宝钢热轧/baogang_hot_data_v4/label_data_v0/hot_jpg", "*/*.json"))
    image_files = [os.path.join(os.path.dirname(anno_file), os.path.basename(anno_file).rsplit(".", 1)[0] + ".jpg") for
                   anno_file in anno_files]

    make_pascal_voc(
        image_anno_path=list(zip(image_files, anno_files)),
        dist_dir="/Users/rensike/Work/宝钢热轧/baogang_hot_data_v4/pascal_voc_hot_vtest",
    )
