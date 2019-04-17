# -*- coding: utf-8 -* -
"""
Pascal Voc格式数据数据增广
"""
import imgaug as ia
from imgaug import augmenters as iaa
from lxml import etree
from utils import dataset_util
from glob import glob
from lxml import etree
from lxml.etree import Element,SubElement,ElementTree
import numpy as np
import cv2
import os
import shutil
import random
import mmcv

def reset_dir(dist_dir):
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    shutil.rmtree(dist_dir)

    dist_image_path = os.path.join(dist_dir, "JPEGImages")
    dist_anno_path = os.path.join(dist_dir, "Annotations")
    dist_imageset_path = os.path.join(dist_dir, "ImageSets", "Main")

    os.makedirs(dist_imageset_path)

    return dist_image_path, dist_anno_path, dist_imageset_path

def _load_bboxes_names(anno_path):
    with open(anno_path, 'rb') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    anno_data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    bboxes = []
    names = []
    if "object" in anno_data:
        for obj in anno_data["object"]:
            xmin = int(obj["bndbox"]["xmin"])
            ymin = int(obj["bndbox"]["ymin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymax = int(obj["bndbox"]["ymax"])
            bboxes.append([xmin,ymin,xmax,ymax])
            names.append(obj["name"])
    return bboxes,names

def _merge_ori_pred(ori_img, pred_img):
    if isinstance(ori_img,str):
        ori_img_np = cv2.imread(ori_img)
    else:
        ori_img_np = ori_img
    merged_img_np = np.concatenate((ori_img_np, pred_img), axis=1)
    return merged_img_np

def map_to_xml(anno_data, dist_path):
    assert "folder" in anno_data
    assert "filename" in anno_data
    assert "size" in anno_data
    assert "width" in anno_data["size"]
    assert "height" in anno_data["size"]
    assert "depth" in anno_data["size"]

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

def pascal_augment(pascal_dir,dist_dir):
    """
    Pascal VOC增广入口
    :param pascal_dir: pascal voc数据目录
    :param dist_dir: 目标生成目录
    :return:
    """
    dist_image_path, dist_anno_path, dist_imageset_path = reset_dir(dist_dir)

    seq_list = [
        iaa.Sequential([iaa.Fliplr(0.5)]),
        iaa.Sequential([iaa.Flipud(0.5)]),
        iaa.Sequential([iaa.Affine(rotate=90)]),
        iaa.Sequential([iaa.Affine(rotate=180)]),
        iaa.Sequential([iaa.Affine(rotate=297),iaa.Multiply((0.7, 1.3))]),
        iaa.Sequential([iaa.Affine(rotate=90),iaa.Multiply((0.7, 1.3))]),
        iaa.Sequential([iaa.Affine(rotate=180),iaa.Multiply((0.7, 1.3))]),
        iaa.Sequential([iaa.Affine(rotate=297),iaa.Multiply((0.7, 1.3))]),
        iaa.Sequential([iaa.Fliplr(0.5),iaa.Affine(rotate=90),iaa.ContrastNormalization()]),
        iaa.Sequential([iaa.Flipud(0.5),iaa.Affine(rotate=180),iaa.ContrastNormalization()]),
        iaa.Sequential([iaa.Fliplr(0.5),iaa.Affine(rotate=297),iaa.ContrastNormalization()]),
        iaa.Sequential([iaa.Flipud(0.5),iaa.Affine(rotate=90),iaa.SigmoidContrast()]),
        iaa.Sequential([iaa.Fliplr(0.5),iaa.Affine(rotate=180),iaa.SigmoidContrast()]),
        iaa.Sequential([iaa.Flipud(0.5),iaa.Affine(rotate=297),iaa.SigmoidContrast()]),
        iaa.Sequential([iaa.Multiply((0.7, 1.3))]),
        iaa.Sequential([iaa.Multiply((0.7, 1.3)),iaa.ContrastNormalization()]),
        iaa.Sequential([iaa.Fliplr(0.5),iaa.Multiply((0.7, 1.3))]),
        iaa.Sequential([iaa.Flipud(0.5),iaa.Multiply((0.7, 1.3))]),
        iaa.Sequential([iaa.Fliplr(0.5),iaa.ContrastNormalization()]),
        iaa.Sequential([iaa.Flipud(0.5),iaa.ContrastNormalization()]),
        iaa.Sequential([iaa.Fliplr(0.5),iaa.Multiply((0.7, 1.3)),iaa.ContrastNormalization()]),
        iaa.Sequential([iaa.Flipud(0.5),iaa.Multiply((0.7, 1.3)),iaa.ContrastNormalization()]),
        iaa.Sequential([iaa.Affine(translate_px={"x": 40, "y": 60},),iaa.ContrastNormalization()]),
        iaa.Sequential([iaa.Affine(translate_px={"x": 40, "y": 60},),iaa.Multiply((0.7, 1.3))]),
        iaa.Sequential([iaa.Affine(translate_px={"x": 40, "y": 60},rotate=90),iaa.Multiply((0.7, 1.3))]),
        iaa.Sequential([iaa.Affine(translate_px={"x": 40, "y": 60},rotate=180),iaa.Multiply((0.7, 1.3))]),
        iaa.Sequential([iaa.Fliplr(0.5),iaa.Affine(translate_px={"x": 40, "y": 60},),iaa.ContrastNormalization()]),
        iaa.Sequential([iaa.Flipud(0.5),iaa.Affine(translate_px={"x": 40, "y": 60},),iaa.Multiply((0.7, 1.3))])
    ]

    trainval_list = mmcv.list_from_file(os.path.join(pascal_dir,"ImageSets/Main","trainval.txt"))
    train_list = mmcv.list_from_file(os.path.join(pascal_dir,"ImageSets/Main","train.txt"))
    val_list = mmcv.list_from_file(os.path.join(pascal_dir,"ImageSets/Main","val.txt"))

    # 在原数据集的基础上增广3倍
    trainval_aug_list = trainval_list + trainval_list + trainval_list

    shutil.copytree(os.path.join(pascal_dir,"JPEGImages"),dist_image_path)
    shutil.copytree(os.path.join(pascal_dir,"Annotations"),dist_anno_path)

    random.shuffle(trainval_aug_list)

    for i,sample in enumerate(trainval_aug_list):
        anno_file = os.path.join(pascal_dir,"Annotations",sample + ".xml")
        image_file = os.path.join(pascal_dir,"JPEGImages",sample + ".jpg")
        image = cv2.imread(image_file)

        if "aug_0_" + sample in trainval_list:
            if "aug_1_" + sample in trainval_list:
                if "aug_2_" + sample in trainval_list:
                    continue
                else:
                    new_sample_name = "aug_2_" + sample
            else:
                new_sample_name = "aug_1_" + sample
        else:
            new_sample_name = "aug_0_" + sample

        trainval_list.append(new_sample_name)
        if sample in train_list:
            train_list.append(new_sample_name)
        if sample in val_list:
            val_list.append(new_sample_name)

        bboxes,names = _load_bboxes_names(anno_file)
        iabbs = []
        for bbox in bboxes:
            iabbs.append(ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))
        bbs = ia.BoundingBoxesOnImage(iabbs, shape=image.shape)

        seq = seq_list[i%len(seq_list)]
        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        new_bbs = bbs_aug.to_xyxy_array().tolist()

        aug_height,aug_width,aug_dim = np.array(image_aug).shape

        anno_data = {
            "folder": dist_image_path,
            "filename": new_sample_name + ".jpg",
            "size": {
                "width": aug_width,
                "height": aug_height,
                "depth": aug_dim
            },
            "objects": []
        }

        for k,bbox in enumerate(new_bbs):
            aug_name = names[k]
            anno_data["objects"].append({
                "name":aug_name,
                "bndbox":{
                    "xmin": int(bbox[0])-1,
                    "ymin": int(bbox[1])-1,
                    "xmax": int(bbox[2])+1,
                    "ymax": int(bbox[3])+1,
                }
            })

        cv2.imwrite(os.path.join(dist_image_path,new_sample_name + ".jpg"), np.array(image_aug))

        dist_xml_write_path = os.path.join(dist_anno_path,new_sample_name + ".xml")
        map_to_xml(anno_data,dist_xml_write_path)

    train_val_set = list(set(trainval_list))
    train_set = list(set(train_list))
    val_set = list(set(val_list))

    with open(os.path.join(dist_imageset_path, "trainval.txt"), "w+") as fw:
        fw.write("\n".join(train_val_set) + "\n")

    with open(os.path.join(dist_imageset_path, "train.txt"), "w+") as fw:
        fw.write("\n".join(train_set) + "\n")

    with open(os.path.join(dist_imageset_path, "val.txt"), "w+") as fw:
        fw.write("\n".join(val_set) + "\n")

    print("write done")


if __name__ == '__main__':
    pascal_augment(
        pascal_dir="/Users/rensike/Work/昆山立讯耳机/blue/pascal_voc_blue_new",
        dist_dir="/Users/rensike/Work/昆山立讯耳机/blue/pascal_voc_blue_new_augment"
    )


