# -*- coding: utf-8 -* -
"""
从pascal框中提取出分类bbox
"""
from utils import dataset_util
from lxml import etree
import os
import shutil
import mmcv
import cv2
import random
from progressbar import ProgressBar

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

def extract_bbox(pascal_dir,dist_path,split="trainval"):
    sample_file = os.path.join(pascal_dir,"ImageSets/Main",split+".txt")
    assert os.path.exists(sample_file)
    if os.path.exists(dist_path):
        shutil.rmtree(dist_path)

    train_path = os.path.join(dist_path,"train")
    test_path = os.path.join(dist_path,"test")
    val_path = os.path.join(dist_path,"val")

    os.makedirs(train_path)
    os.makedirs(test_path)
    os.makedirs(val_path)

    max_train_count_each_class = 2100
    max_test_count_each_class = 600
    max_val_count_each_class = 300
    sample_list = mmcv.list_from_file(sample_file)
    random.shuffle(sample_list)
    label_cnt = {}
    total_size = {}

    pbar = ProgressBar(maxval=len(sample_list))
    pbar.start()

    for ix,sample in enumerate(sample_list):
        pbar.update(ix)
        image = cv2.imread(os.path.join(pascal_dir,"JPEGImages",sample+".jpg"))
        height,width = image.shape[:2]
        anno_data = _load_anno_sample(os.path.join(pascal_dir,"Annotations",sample+".xml"))
        if "object" not in anno_data or len(anno_data["object"]) == 0:
            continue

        for obj in anno_data["object"]:
            ymin,xmin,ymax,xmax = int(obj["bndbox"]["ymin"]), \
                                  int(obj["bndbox"]["xmin"]), \
                                  int(obj["bndbox"]["ymax"]),\
                                  int(obj["bndbox"]["xmax"])

            # 对bbox随机偏移
            if random.random() > 0.7:
                x_w = (xmax - xmin) / 2
                y_h = (ymax - ymin) / 2

                y_shift = (random.random() * 2 - 1) * y_h
                x_shift = (random.random() * 2 - 1) * x_w

                ymin += y_shift
                ymax += y_shift
                xmin += x_shift
                xmax += x_shift

                ymin = round(max(1, ymin))
                xmin = round(max(1, xmin))

                ymax = round(min(height, ymax))
                xmax = round(min(width, xmax))

            # print("xmin:{},ymin:{},xmax:{},ymax:{}".format(xmin,ymin,xmax,ymax))

            label = obj["name"]
            if label not in label_cnt:
                label_cnt[label] = 0

            if label_cnt[label] < max_train_count_each_class:
                _dist_path = train_path
            elif label_cnt[label] < max_train_count_each_class + max_val_count_each_class:
                _dist_path = val_path
            else:
                _dist_path = test_path

            save_dir = os.path.join(_dist_path,label)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if label_cnt[label] > (max_train_count_each_class+max_test_count_each_class+max_val_count_each_class):
                continue

            if label not in total_size:
                total_size[label] = (0,0)

            total_size[label] = (total_size[label][0]+ymax-ymin,total_size[label][1]+xmax-xmin)

            save_path = os.path.join(save_dir,"{}_{}.jpg".format(label,label_cnt[label]))
            label_cnt[label] += 1

            cv2.imwrite(save_path,image[ymin:ymax,xmin:xmax,:])

    pbar.finish()
    print("done,",label_cnt)
    print("avg size:")
    for label in total_size:
        print("{} avg height:{}".format(label,total_size[label][0] / label_cnt[label]))
        print("{} avg width:{}".format(label,total_size[label][1] / label_cnt[label]))


if __name__ == '__main__':
    extract_bbox(
        pascal_dir="/Users/rensike/Work/宝钢热轧/baogang_hot_data_v4/pascal_voc_hot_v1",
        dist_path="/Users/rensike/Work/宝钢热轧/baogang_hot_data_v4/hot_bbox_class_v1"
    )


