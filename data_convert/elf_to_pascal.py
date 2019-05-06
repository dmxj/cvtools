# -*- coding: utf-8 -* -
"""
标注精灵转 pascal voc
"""

import os
import shutil
import glob
from lxml import etree
from lxml.etree import Element,SubElement,ElementTree
from utils import anno_util
import cv2
import random

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


def make_pascal_voc(source_dir, dist_dir,suffix="jpg"):
    """
    :param source_dir: 数据源目录
    :param dist_dir: Pascal数据生成目录
    :param suffix: 图片后缀
    :return:
    """
    dist_image_path, dist_anno_path, dist_imageset_path = reset_dir(dist_dir)
    classes = os.listdir(source_dir)

    trainval_list = []
    for cls in classes:
        anno_files = glob.glob(os.path.join(source_dir, cls, "outputs", "*.xml"))
        for anno_file in anno_files:
            anno_name = os.path.basename(anno_file)
            image_file = os.path.join(source_dir, cls, "{}.{}".format(anno_name.rsplit(".", 1)[0],suffix))
            with open(anno_file, 'r') as fid:
                xml_str = fid.read()
            if "<bndbox>" not in xml_str:
                continue
            xml = etree.fromstring(xml_str)
            xml_data = anno_util.recursive_parse_xml_to_dict(xml)
            im_raw = cv2.imread(image_file)
            height, width, dim = im_raw.shape
            anno_data = {
                "folder": dist_image_path,
                "filename": os.path.basename(image_file),
                "size": {
                    "width": width,
                    "height": height,
                    "depth": dim
                },
                "objects": []
            }

            if "annotation" in xml_data:
                for object in xml_data["annotation"]["object"]:
                    anno_data["objects"].append({
                        "name": object["name"],
                        "bndbox": object["bndbox"]
                    })
            elif "doc" in xml_data:
                for object in xml_data["doc"]["outputs"]["object"]:
                    anno_data["objects"].append({
                        "name": object["item"]["name"],
                        "bndbox": object["item"]["bndbox"]
                    })
            else:
                continue

            dist_anno_file = os.path.join(dist_anno_path, anno_name)
            map_to_xml(anno_data, dist_anno_file)
            shutil.copyfile(image_file,os.path.join(dist_image_path,os.path.basename(image_file)))
            trainval_list.append(str(anno_name.rsplit(".", 1)[0]))

        print("write {} class done.".format(cls))

    train_list = random.sample(trainval_list, int(len(trainval_list) * 0.8))
    val_list = [i for i in trainval_list if i not in train_list]

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
        dist_dir=""
    )



