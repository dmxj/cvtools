# -*- coding: utf-8 -* -
"""
针对pascal voc数据的大图切小图
"""
import os
import mmcv
import glob
import shutil
import cv2
import copy
import shapely
import json
from shapely.geometry import Polygon,MultiPoint
import numpy as np
from utils import dataset_util
from lxml import etree
from lxml.etree import Element,SubElement,ElementTree

def _load_anno_sample(anno_file):
    """
    加载标注图片
    :param anno_file:
    :return:
    """
    with open(anno_file, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    anno_data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    return anno_data

def _load_labelme_points_list(labelme_anno_file):
    """
    加载labelme标注文件中的polygon列表
    :param labelme_anno_file:
    :return:
    """
    anno_data = json.load(open(labelme_anno_file,"r"))
    points_list = [shape["points"] for shape in anno_data["shapes"]]
    return points_list

def _reset_dir(dist_dir,is_seg=False):
    """
    创建pascal voc数据集的目录
    :param dist_dir:
    :return:
    """
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    shutil.rmtree(dist_dir)

    dist_image_path = os.path.join(dist_dir, "JPEGImages")
    dist_anno_path = os.path.join(dist_dir, "Annotations")
    if is_seg:
        dist_imageset_path = os.path.join(dist_dir, "ImageSets", "Segmentation")
    else:
        dist_imageset_path = os.path.join(dist_dir, "ImageSets", "Main")
    dist_segclass_path = os.path.join(dist_dir, "SegmentationClass")
    dist_segobj_path = os.path.join(dist_dir, "SegmentationObject")

    os.makedirs(dist_image_path)
    os.makedirs(dist_anno_path)
    os.makedirs(dist_imageset_path)
    os.makedirs(dist_segclass_path)
    os.makedirs(dist_segobj_path)

    return dist_image_path, dist_anno_path, dist_imageset_path, dist_segclass_path,dist_segobj_path

def _get_bbox_from_points(points):
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

    return xmin,ymin,xmax,ymax


def cut_image(image_path,cut_width=300,cut_height=300,overlap=120):
    """
    切图，并返回小图的列表
    :param image_path:
    :param cut_width:
    :param cut_height:
    :param overlap:
    :return:
    """
    img = cv2.imread(image_path)
    height, width, channel = img.shape

    xmin = 0
    ymin = 0

    cut_image_list = []

    while xmin < width - overlap and ymin < height - overlap:
        while xmin < width - overlap:
            ymin = 0
            if xmin + cut_width < width:
                xmax = xmin + cut_width
            else:
                xmax = width
            while ymin < height - overlap:
                if ymin + cut_height < height:
                    ymax = ymin + cut_height
                else:
                    ymax = height

                cut_image_list.append(
                    (
                        img[ymin:ymax, xmin:xmax],xmin,ymin
                    )
                )

                ymin = ymin + cut_height - overlap
            xmin = xmin + cut_width - overlap

    return cut_image_list

def calc_cut_image_bbox_with_polygon(polygon_list,width,height,x_offset,y_offset):
    """
    根据polygon获取切图的bouding box坐标信息
    :param polygon_list:list of polygon,each format: [[x1,y1],[x2,y2],...]
    :param width:
    :param height:
    :param x_offset:
    :param y_offset:
    :return:
    """
    if len(polygon_list) == 0:
        return []

    bbox_list = []
    for polygon in polygon_list:
        if len(polygon) == 0:
            continue
        origin_pg = Polygon(np.array(polygon))
        cut_pg = Polygon(np.array([[x_offset,y_offset],[x_offset,y_offset+height],[x_offset+width,y_offset+height],[x_offset+width,y_offset]]))

        intersec_hull = origin_pg.intersection(cut_pg).convex_hull
        try:
            coords = list(intersec_hull.exterior.coords)
            if len(coords) <= 2:
                continue

            xmin, ymin, xmax, ymax = _get_bbox_from_points(coords)
            xmin = max(0,xmin-x_offset)
            ymin = max(0,ymin-y_offset)
            xmax = max(0,xmax-x_offset)
            ymax = max(0,ymax-y_offset)

            bbox_list.append([xmin,ymin,xmax,ymax])
        except:
            continue

    return bbox_list

def calc_cut_image_bbox(origin_anno_data,width,height,x_offset,y_offset):
    """
    获取切图的新的bounding box坐标信息
    :param origin_anno_data:
    :param x_offset:
    :param y_offset:
    :return:
    """
    if origin_anno_data is None or "object" not in origin_anno_data or len(origin_anno_data["object"]) < 1:
        return []

    cut_object_infos = []
    for object in origin_anno_data["object"]:
        origin_x_min = int(object["bndbox"]["xmin"])
        origin_y_min = int(object["bndbox"]["ymin"])
        origin_x_max = int(object["bndbox"]["xmax"])
        origin_y_max = int(object["bndbox"]["ymax"])
        if origin_x_max <= x_offset or origin_x_min >= x_offset + width or origin_y_max <= y_offset or origin_y_min \
                >= y_offset + height:
            continue

        if origin_x_min < x_offset:
            origin_x_min = x_offset
        if origin_y_min < y_offset:
            origin_y_min = y_offset
        if origin_x_max > x_offset + width:
            origin_x_max = x_offset + width
        if origin_y_max > y_offset + height:
            origin_y_max = y_offset + height

        xmin = origin_x_min - x_offset
        ymin = origin_y_min - y_offset
        xmax = origin_x_max - x_offset
        ymax = origin_y_max - y_offset

        cut_object = copy.deepcopy(object)
        cut_object["bndbox"]["xmin"] = xmin
        cut_object["bndbox"]["ymin"] = ymin
        cut_object["bndbox"]["xmax"] = xmax
        cut_object["bndbox"]["ymax"] = ymax
        cut_object_infos.append(cut_object)

    return cut_object_infos

def write_anno_xml(anno_data, anno_file_path):
    """
    将标注信息按照pascal格式写入xml文件
    :param anno_data:
    :param anno_file_path:
    :return:
    """
    assert "folder" in anno_data
    assert "filename" in anno_data
    assert "size" in anno_data
    assert "segmented" in anno_data
    assert "width" in anno_data["size"]
    assert "height" in anno_data["size"]
    assert "depth" in anno_data["size"]
    assert "objects" in anno_data

    segmented = anno_data["segmented"]

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
        SubElement(obj, 'pose').text = object["pose"]
        SubElement(obj, 'truncated').text = object["truncated"]
        SubElement(obj, 'difficult').text = object["difficult"]
        bndbox = SubElement(obj, 'bndbox')
        SubElement(bndbox, 'xmin').text = str(object["bndbox"]["xmin"])
        SubElement(bndbox, 'ymin').text = str(object["bndbox"]["ymin"])
        SubElement(bndbox, 'xmax').text = str(object["bndbox"]["xmax"])
        SubElement(bndbox, 'ymax').text = str(object["bndbox"]["ymax"])

    tree = ElementTree(root)
    tree.write(anno_file_path, encoding='utf-8',pretty_print=True)

def pascal_crop(pascal_voc_dir,label_me_dir,dist_dir,is_seg=False,suffix="jpg",cut_width=300,cut_height=300,overlap=120):
    """
    切图入口函数
    :param pascal_voc_dir:
    :param label_me_dir:
    :param dist_dir:
    :param is_seg:
    :param suffix:
    :param cut_width:
    :param cut_height:
    :param overlap:
    :return:
    """
    dist_image_path, dist_anno_path, dist_imageset_path, \
    dist_segclass_path, dist_segobj_path = _reset_dir(dist_dir=dist_dir,is_seg=is_seg)
    if is_seg:
        train_sample_list = mmcv.list_from_file(os.path.join(pascal_voc_dir,"ImageSets/Segmentation","train.txt"))
        val_sample_list = mmcv.list_from_file(os.path.join(pascal_voc_dir,"ImageSets/Segmentation","val.txt"))
    else:
        train_sample_list = mmcv.list_from_file(os.path.join(pascal_voc_dir,"ImageSets/Main","train.txt"))
        val_sample_list = mmcv.list_from_file(os.path.join(pascal_voc_dir,"ImageSets/Main","val.txt"))

    cut_train_sample_list = []
    cut_val_sample_list = []
    for sample_name in list(set(train_sample_list + val_sample_list)):
        image_file = os.path.join(pascal_voc_dir,"JPEGImages","{}.{}".format(sample_name,suffix))
        anno_file = os.path.join(pascal_voc_dir,"Annotations",sample_name + ".xml")
        anno_data = _load_anno_sample(anno_file)

        if label_me_dir is not None:
            labelme_anno_file = os.path.join(label_me_dir, sample_name + ".json")
            points_list = _load_labelme_points_list(labelme_anno_file)

        cut_image_list = cut_image(image_file,cut_width,cut_height,overlap)
        contain_target_list = []

        for cut_item in cut_image_list:
            (cut_image_np,x_offset,y_offset) = cut_item
            cut_sample_name = "{}_{}_{}".format(sample_name,x_offset,y_offset)

            if sample_name in train_sample_list:
                cut_train_sample_list.append(cut_sample_name)
            if sample_name in val_sample_list:
                cut_val_sample_list.append(cut_sample_name)

            cut_image_save_path = os.path.join(dist_image_path,"{}.{}".format(cut_sample_name,suffix))
            cv2.imwrite(cut_image_save_path, cut_image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            cut_image_height,cut_image_width,dim = cut_image_np.shape
            cut_anno_data = copy.deepcopy(anno_data)

            if label_me_dir is None:
                cut_object_infos = calc_cut_image_bbox(anno_data, cut_image_width, cut_image_height, x_offset, y_offset)
                cut_anno_data["objects"] = cut_object_infos
                if len(cut_object_infos) == 0:  # 如果不包含检测目标则没有分割的数据
                    cut_anno_data["segmented"] = "0"
                else:  # 包含检测目标的切图样本
                    contain_target_list.append(cut_sample_name)
            else:
                cut_bbox_list = calc_cut_image_bbox_with_polygon(points_list,cut_image_width,cut_image_height,x_offset,y_offset)
                cut_anno_data["objects"] = [{
                    "name":"huahen",
                    "pose":"Unspecified",
                    "truncated":"0",
                    "difficult":"0",
                    "bndbox":{
                        "xmin":str(int(bbox[0])),
                        "ymin":str(int(bbox[1])),
                        "xmax":str(int(bbox[2])),
                        "ymax":str(int(bbox[3])),
                    }
                } for bbox in cut_bbox_list]
                if len(cut_bbox_list) == 0: # 如果不包含检测目标则没有分割的数据
                    cut_anno_data["segmented"] = "0"
                else: # 包含检测目标的切图样本
                    contain_target_list.append(cut_sample_name)


            cut_anno_data["folder"] = dist_image_path
            cut_anno_data["filename"] = os.path.basename(cut_image_save_path)
            cut_anno_data["size"]["width"] = cut_image_width
            cut_anno_data["size"]["height"] = cut_image_height

            cut_anno_save_path = os.path.join(dist_anno_path,cut_sample_name + ".xml")
            write_anno_xml(cut_anno_data,cut_anno_save_path)

        if is_seg and int(anno_data["segmented"]) == 1:
            segment_origin_image_files = [
                os.path.join(pascal_voc_dir,"SegmentationClass",sample_name + ".png"),
                os.path.join(pascal_voc_dir, "SegmentationObject", sample_name + ".png")
            ]

            for i,segment_image_file in enumerate(segment_origin_image_files):
                if not os.path.exists(segment_image_file):
                    continue
                dis_seg_image_path = dist_segclass_path if i == 0 else dist_segobj_path
                cut_image_list = cut_image(segment_image_file, cut_width, cut_height, overlap)
                for (cut_image_np,x_offset,y_offset) in cut_image_list:
                    # 如果当前的分割切图包含检测目标则保存对应的分割mask图片
                    if "{}_{}_{}".format(sample_name,x_offset,y_offset) not in contain_target_list:
                        continue
                    cv2.imwrite(
                        filename=os.path.join(dis_seg_image_path,"{}_{}_{}.png".format(sample_name,x_offset,y_offset)),
                        img=cut_image_np,
                        params=[int(cv2.IMWRITE_JPEG_QUALITY), 100])

        print("cut {} done.".format(sample_name))

    cut_trainval_sample_list = list(set(cut_train_sample_list + cut_val_sample_list))
    with open(os.path.join(dist_imageset_path,"train.txt"),"w") as fw:
        fw.write("\n".join(cut_train_sample_list) + "\n")
    with open(os.path.join(dist_imageset_path,"val.txt"),"w") as fw:
        fw.write("\n".join(cut_val_sample_list) + "\n")
    with open(os.path.join(dist_imageset_path,"trainval.txt"),"w") as fw:
        fw.write("\n".join(cut_trainval_sample_list) + "\n")

    print("done.")

if __name__ == '__main__':
    pascal_crop(
        pascal_voc_dir="/Users/rensike/Work/昆山立讯耳机/liaohuaqipao/pascal_voc_data_white",
        dist_dir="/Users/rensike/Work/昆山立讯耳机/liaohuaqipao/pascal_voc_data_liaohuaqipao_cuted_3_3",
        # label_me_dir="/Users/rensike/Work/昆山立讯耳机/labels",
        label_me_dir=None,
        is_seg=False,
        suffix="jpg",
        # cut_width=738,
        # cut_height=576,
        # overlap=120,
        cut_width=964,
        cut_height=748,
        overlap=150
    )


