# -*- coding: utf-8 -* -
"""
coco格式数据转Pascal
"""
import os
import json
import shutil
import cv2
from lxml.etree import Element,SubElement,ElementTree

def _list_from_file(filename, prefix='', offset=0, max_num=0):
    cnt = 0
    item_list = []
    with open(filename, 'r') as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num > 0 and cnt >= max_num:
                break
            item_list.append(prefix + line.rstrip('\n'))
            cnt += 1
    return item_list

def _list_to_file(list_data, filename, prefix=''):
    with open(filename, 'w') as f:
        f.write("\n".join([prefix+str(i) for i in list_data]) + "\n")

def _is_dir(dir):
    return os.path.isdir(dir) and os.path.exists(dir)

def _reset_dir(dist_dir):
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

def _map_to_xml(anno_data, dist_path):
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

def make_trainval_set(dist_pascal_root):
    dist_imageset_path = os.path.join(dist_pascal_root, "ImageSets", "Main")
    assert _is_dir(dist_pascal_root) and _is_dir(dist_imageset_path)
    samples = []
    for s in os.listdir(dist_imageset_path):
        samples = samples + _list_from_file(dist_imageset_path + "/" + s)
    samples = [_ for _ in samples if _ != ""]
    _list_to_file(samples,dist_imageset_path+"/trainval.txt")

def coco_to_voc(image_dir,anno_path,dist_dir,split="train",reset=False):
    if reset:
        dist_image_path, dist_anno_path, dist_imageset_path = _reset_dir(dist_dir)
    else:
        dist_image_path = os.path.join(dist_dir, "JPEGImages")
        dist_anno_path = os.path.join(dist_dir, "Annotations")
        dist_imageset_path = os.path.join(dist_dir, "ImageSets", "Main")

    assert _is_dir(dist_image_path) and _is_dir(dist_anno_path) and _is_dir(dist_imageset_path)

    anno_data = json.load(open(anno_path,"r"))
    images = anno_data["images"]
    annos = anno_data["annotations"]
    categories = anno_data["categories"]
    id2cat = {cate["id"]:cate["name"] for cate in categories}
    anno_dict = {}

    for anno in annos:
        if anno["image_id"] not in anno_dict:
            anno_dict[anno["image_id"]] = []
        anno_dict[anno["image_id"]].append(anno)

    samples = []
    for image in images:
        sample_name = image["file_name"].rsplit(".",1)[0]
        samples.append(sample_name)
        image_file = os.path.join(image_dir,image["file_name"])
        annos = anno_dict[image["id"]]
        shutil.copyfile(image_file,os.path.join(dist_image_path,image["file_name"]))

        img = cv2.imread(image_file)
        height,width,dim = img.shape
        voc_anno = {
            "segmented": "1" if "segmentation" in annos[0] and len(annos[0]["segmentation"]) > 0 else "0",
            "folder": dist_image_path,
            "filename": image["file_name"],
            "size": {
                "width": width,
                "height": height,
                "depth": dim
            },
            "objects": []
        }

        for anno in annos:
            voc_anno["objects"].append({
                "name": id2cat[int(anno["category_id"])],
                "bndbox": {
                    "xmin":round(anno["bbox"][0]),
                    "ymin":round(anno["bbox"][1]),
                    "xmax":round(anno["bbox"][0]+anno["bbox"][2]),
                    "ymax":round(anno["bbox"][1]+anno["bbox"][3]),
                }
            })

        dist_anno_file = os.path.join(dist_anno_path, "{}.xml".format(sample_name))
        _map_to_xml(voc_anno, dist_anno_file)

        with open(os.path.join(dist_imageset_path, "%s.txt" % split), "w+") as fw:
            fw.write("\n".join(samples) + "\n")

if __name__ == '__main__':
    # coco train data to pascal
    coco_to_voc(
        image_dir="/Users/rensike/Files/temp/coco_tiny/train",
        anno_path="/Users/rensike/Files/temp/coco_tiny/annotations/instances_train.json",
        dist_dir="/Users/rensike/Files/temp/coco_tiny_to_voc",
        split="train",
        reset=True
    )

    # coco test data to pascal
    coco_to_voc(
        image_dir="/Users/rensike/Files/temp/coco_tiny/test",
        anno_path="/Users/rensike/Files/temp/coco_tiny/annotations/instances_test.json",
        dist_dir="/Users/rensike/Files/temp/coco_tiny_to_voc",
        split="test",
    )

    # coco val data to pascal
    coco_to_voc(
        image_dir="/Users/rensike/Files/temp/coco_tiny/val",
        anno_path="/Users/rensike/Files/temp/coco_tiny/annotations/instances_val.json",
        dist_dir="/Users/rensike/Files/temp/coco_tiny_to_voc",
        split="val",
    )

    # make pascal trainval
    make_trainval_set("/Users/rensike/Files/temp/coco_tiny_to_voc")