# -*- coding: utf-8 -* -
"""
统计图片的均值、方差
"""
import cv2
import numpy as np

def get_mean_std(image_files):
    """
    获取图片均值方差
    :param image_files:
    :return:
    """
    ims = [cv2.imread(image_file) for image_file in image_files]
    shape = ims[0].shape

    if len(shape) < 3:
        (mean, stddv) = cv2.meanStdDev(np.array(ims))
        return mean[0],stddv[0]
    else:
        im_mean = []
        im_std = []
        for dim in range(shape[2]):
            (mean, stddv) = cv2.meanStdDev(np.array(ims)[:, :, :, dim])
            im_mean.append(mean[0][0])
            im_std.append(stddv[0][0])
        return im_mean,im_std

def get_avg_width_height(image_files):
    """
    获取图像平均宽高
    :param image_files:
    :return:
    """
    height_list = []
    width_list = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        height,width,_ = img.shape
        height_list.append(height)
        width_list.append(width)

    return sum(height_list)/len(height_list),sum(width_list)/len(width_list)

if __name__ == '__main__':
    from glob import glob

    image_files = glob("/Users/rensike/Work/jiepu/dimian/VOC2012_no_bg/JPEGImages/*.jpg")

    avg_height,avg_width = get_avg_width_height(image_files)
    print("avg image height:",avg_height)
    print("avg image width:",avg_width)

    mean,std = get_mean_std(image_files)
    print("image mean:",mean)
    print("image std:",std)
