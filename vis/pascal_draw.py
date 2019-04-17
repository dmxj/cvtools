# -*- coding: utf-8 -* -
"""
Pascal Voc数据绘制
"""
from utils import dataset_util
from utils import voc_mask_util
from utils import visualization_utils as vis_util
from utils import image_util
from utils import label_map_util
from PIL import Image
from lxml import etree
import numpy as np
import glob
import os
import cv2
import shutil

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

def _get_mask_np(mask_image_path,bbox):
    """
    根据mask png分割图片返回mask数组
    :param mask_image_path: 分割的mask png图片文件路径
    :param bbox: 分割框列表
    :return:
    """
    seg_image_mask = cv2.imread(mask_image_path, 0)
    mask_list = [np.expand_dims(voc_mask_util.getsegmask(seg_image_mask,[box[1],box[0],box[3],box[2]]),0) for box in list(bbox)]
    return np.vstack(mask_list)

def _load_label_map(label_map_path):
    '''
    加载label，生成 {类id : 类名} 的映射字典
    :return:
    '''
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
    category_name_index = {category_index[cat_id]["name"]: category_index[cat_id] for cat_id in category_index}
    categories = label_map_util.create_categories_from_labelmap(label_map_path)

    return category_index,category_name_index,categories


def draw_and_save_gt(source_dir,output_dir,label_map_path,only_target=False):
    image_files = glob.glob(os.path.join(source_dir,"JPEGImages","*.jpg"))
    category_index, category_name_index, categories = _load_label_map(label_map_path)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    count = 0
    for image_path in image_files:
        sample_name = os.path.basename(image_path).rsplit(".", 1)[0]
        anno_path = os.path.join(source_dir,"Annotations","{}.xml".format(sample_name))
        anno_data = _load_anno_sample(anno_path)

        boxes = []
        classes = []
        scores = []

        if only_target:
            if "object" not in anno_data or len(anno_data["object"]) == 0:
                continue

        if "object" in anno_data:
            for obj in anno_data["object"]:
                boxes.append(
                    [int(obj["bndbox"]["ymin"]), int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymax"]),
                     int(obj["bndbox"]["xmax"])])
                classes.append(category_name_index[obj["name"]]["id"])
                scores.append(1.0)
        boxes = np.array(boxes)
        classes = np.array(classes)
        scores = np.array(scores)

        mask_np = None

        if int(anno_data["segmented"]) == 1:
            mask_image_path = os.path.join(source_dir,"SegmentationObject","{}.png".format(sample_name))
            mask_np = _get_mask_np(mask_image_path, boxes)
            mask_np[mask_np!=0] = 1

            print("image_path:",image_path)
            print("anno_path:",anno_path)
            print("mask_image_path:",mask_image_path)

        image_np = cv2.imread(image_path)
        if image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            classes,
            scores,
            category_index,
            instance_masks=mask_np,
            use_normalized_coordinates=False,
            line_thickness=1)
        image_drawed = image_util.load_numpy_array_into_image(image_np)

        save_path = os.path.join(output_dir,os.path.basename(image_path))
        image_drawed.save(save_path)

        count += 1

    print("draw done, count is: ",count)

if __name__ == '__main__':
    draw_and_save_gt(
        source_dir="",
        output_dir="",
        label_map_path=""
    )

