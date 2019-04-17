# -*- coding: utf-8 -* -
import numpy as np
from PIL import Image


def load_image_into_numpy_array(image):
    '''
    将图片加载为numpy数组
    :param image: PIL图片
    :return:
    '''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_numpy_array_into_image(np_arr):
    '''
    将numpy array转换为图片
    :param np_arr:
    :return:
    '''
    im = Image.fromarray(np_arr.astype('uint8')).convert('RGB')
    return im

def concat_images(image_list,is_show=False,save_path=None):
    '''
    将多张图横向拼接成一张图
    :param image_list: 图片路径列表或图片矩阵列表
    :param is_show:
    :param save_path:
    :return:
    '''
    if isinstance(image_list[0],str):
        ims = [Image.open(img_file) for img_file in image_list]
    else:
        ims = image_list

    total_width = 0
    max_height = 0

    for im in ims:
        width, height = im.size
        total_width += width
        if height > max_height:
            max_height = height

    result = Image.new(ims[0].mode, (total_width, max_height))

    x = 0
    for i, im in enumerate(ims):
        width, height = im.size
        result.paste(im, box=(x, 0))
        x += width

    if is_show:
        result.show()

    if save_path is not None:
        result.save(save_path)