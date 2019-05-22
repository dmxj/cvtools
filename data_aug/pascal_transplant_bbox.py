# -*- coding: utf-8 -* -
"""
Pascal Voc 拷贝移植bbox进行数据增广
"""
import shutil
import os
from lxml import etree
from lxml.etree import Element,SubElement,ElementTree
from lxml import etree
from utils import dataset_util
from glob import glob
import numpy as np
import cv2
import os
import shutil
import random
import mmcv

def _reset_dir(dist_dir):
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

def _is_point_in(x,y,bbox):
    """
    判断某个点是否在bbox范围内
    :param x:
    :param y:
    :param bbox:
    :return:
    """
    bbox = bbox.tolist() if isinstance(bbox,np.ndarray) else bbox
    assert len(bbox) == 4
    return x>=bbox[0] and y>=bbox[1] and x<=bbox[2] and y<=bbox[3]

def _transplant_bbox(bboxes,target_image,target_bboxes):
    dist_img = mmcv.imread(target_image)
    dist_height,dist_width = dist_img.shape[:2]
    if isinstance(bboxes,list):
        bboxes = np.array(bboxes)
    assert len(bboxes.shape) == 2 and bboxes.shape[1] == 4

    if isinstance(target_bboxes,list):
        target_bboxes = np.array(target_bboxes)
    assert len(target_bboxes.shape) == 2 and target_bboxes.shape[1] == 4

    target_mask = np.zeros((dist_height,dist_width))
    for i in target_bboxes.shape[0]:
        target_bbox = target_bboxes[i,:]
        target_mask[target_bbox[1]:target_bbox[3],target_bbox[0]:target_bbox[2]] = 1

    # 科学选择，待实现
    success_bboxes = []
    for i in bboxes.shape[0]:
        bbox = bboxes[i,:]
        blank_area = np.where(target_mask == 0)
        m = random.randint(0,blank_area[0].shape[0]-1)
        xmin,ymin = blank_area[0][m],blank_area[1][m]
        xmax = xmin + bbox[2] - bbox[0]
        ymax = ymin + bbox[3] - bbox[1]

        if target_mask[xmin,ymin] == 1 or target_mask[xmin,ymax] == 1 or target_mask[xmax,ymin] == 1 or target_mask[xmax,ymax] == 1:
            success_bboxes.append([xmin,ymin,xmax,ymax])

    return success_bboxes

def transplant_bbox(pascal_dir,dist_dir,max_count=1000):
    """
    拷贝、移植bbox入口
    :param pascal_dir:
    :param dist_dir:
    :return:
    """
    dist_image_path, dist_anno_path, dist_imageset_path = _reset_dir(dist_dir)
    origin_samples = mmcv.list_from_file(os.path.join(pascal_dir,"ImageSets/Main/trainval.txt"))
    origin_samples = [_ for _ in origin_samples if "hongtiepi" not in _]

    samples = random.sample(origin_samples,max_count)

    cat_count = {"heixian":0,"heitiao":0,"huashang":0,"hongtiepi":0}
    for sample in samples:
        anno_file = os.path.join(pascal_dir,"Annotations","%s.xml" % sample)
        image_file = os.path.join(pascal_dir,"JPEGImages","%s.jpg" % sample)
        if "heixian" in sample:
            cat = "heixian"
        elif "heitiao" in sample:
            cat = "heitiao"
        elif "huashang" in sample:
            cat = "huashang"
        elif "hongtiepi" in sample:
            cat = "hongtiepi"
        else:
            continue

        new_sample = "{}_cp_{}".format(cat,cat_count[cat])
        cat_count[cat] += 1
        bboxes, names = _load_bboxes_names(anno_file)
        cnt = max(1,int(len(bboxes)/2))
        bboxes = random.sample(bboxes,cnt)

        target_sample = random.choice(samples)
        if target_sample == sample:
            continue
        target_anno_file = os.path.join(pascal_dir,"Annotations","%s.xml" % target_sample)
        target_image_file = os.path.join(pascal_dir,"JPEGImages","%s.jpg" % target_sample)

        target_bboxes,_ = _load_bboxes_names(target_anno_file)
        _transplant_bbox(bboxes, target_image_file, np.array(target_bboxes))










