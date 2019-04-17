# -*- coding: utf-8 -* -
'''
Pascal Voc 分割图片相关操作
'''
import cv2
import numpy as np
from PIL import Image, ImageDraw


def getsegmentation(seg_img_path, bbox):
    '''
    获取分割图片区域的mask（pascal voc数据中，每个分割区域都有一个检测框与之对应）
    :param seg_img_path: 分割的png图片地址
    :param bbox: 分割区域对应的框的坐标[xmin ymin xmax ymax]
    :return:
    '''
    try:
        mask_1 = cv2.imread(seg_img_path, 0)
        mask = np.zeros_like(mask_1, np.uint8)
        rectangle = bbox
        mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                     rectangle[0]:rectangle[2]]

        # 计算矩形中点像素值
        mean_x = (rectangle[0] + rectangle[2]) // 2
        mean_y = (rectangle[1] + rectangle[3]) // 2

        end = min((mask.shape[1], int(rectangle[2]) + 1))
        start = max((0, int(rectangle[0]) - 1))

        flag = True
        for i in range(mean_x, end):
            x_ = i
            y_ = mean_y
            pixels = mask_1[y_, x_]
            if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                mask = (mask == pixels).astype(np.uint8)
                flag = False
                break
        if flag:
            for i in range(mean_x, start, -1):
                x_ = i
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:
                    mask = (mask == pixels).astype(np.uint8)
                    break
        return mask

    except Exception as e:
        print("getsegmentation error:",e)
        return [0]

def getsegmask(seg_img_mask, bbox):
    '''
    获取分割图片区域的mask列表（pascal voc数据中，每个分割区域都有一个检测框与之对应）
    :param seg_img_mask: 分割的png图片对应的numpy数组，使用cv2.read("/path/to/image",0)
    :param bbox: 分割区域对应的框的坐标列表[[xmin1 ymin1 xmax1 ymax1],[xmin2 ymin2 xmax2 ymax2],...]
    :return:
    '''
    try:
        mask = np.zeros_like(seg_img_mask, np.uint8)
        rectangle = bbox
        mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = seg_img_mask[rectangle[1]:rectangle[3],
                                                                     rectangle[0]:rectangle[2]]

        # 计算矩形中点像素值
        mean_x = (rectangle[0] + rectangle[2]) // 2
        mean_y = (rectangle[1] + rectangle[3]) // 2

        end = min((mask.shape[1], int(rectangle[2]) + 1))
        start = max((0, int(rectangle[0]) - 1))

        flag = True
        for i in range(mean_x, end):
            x_ = i
            y_ = mean_y
            pixels = seg_img_mask[y_, x_]
            if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                mask = (mask == pixels).astype(np.uint8)
                flag = False
                break
        if flag:
            for i in range(mean_x, start, -1):
                x_ = i
                y_ = mean_y
                pixels = seg_img_mask[y_, x_]
                if pixels != 0 and pixels != 220:
                    mask = (mask == pixels).astype(np.uint8)
                    break
        return mask

    except Exception as e:
        print("getsegmentation error:",e)
        return [0]

def mask2box(mask):
    '''
    从mask反算出其边框
    mask：[h,w]  0、1组成的图片
    1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
    :param mask:
    :return:
    '''
    # np.where(mask==1)
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    # 解析左上角行列号
    left_top_r = np.min(rows)  # y
    left_top_c = np.min(clos)  # x

    # 解析右下角行列号
    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)

    # [xmin,ymin,xmax,ymax]
    return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]


def mask2polygons(mask):
    '''
    根据分割区域的mask计算对应的轮廓坐标点
    :param mask:
    :return:
    '''
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
    bbox = []
    for cont in contours[1]:
        cont_list = list(cont.flatten())
        for i in range(0,len(cont_list),2):
            # [x坐标,y坐标]
            bbox.append([cont_list[i],cont_list[i+1]])
        # [bbox.append(i) for i in list(cont.flatten())]
        # map(bbox.append,list(cont.flatten()))
    return bbox  # list(contours[1][0].flatten())

def polygons_to_mask(img_shape, polygons):
    '''
    根据图片的shape，以及每个分割区域的polygon获取mask
    :param img_shape:
    :param polygons:
    :return:
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = Image.fromarray(mask)
    # xy = list(map(tuple, polygons))
    xy = [tuple(p) for p in polygons]
    ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def show_mask(img_path,polygons):
    '''
    显示分割区域
    :param img_path:
    :param polygons:
    :return:
    '''
    img = cv2.imread(img_path)
    height,width,_ = img.shape

    img = np.zeros([height,width],np.uint8)
    cv2.polylines(img, [np.asarray(polygons)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
    cv2.fillPoly(img, [np.asarray(polygons)], 200)  # 画多边形 内部像素值为1

    cv2.imshow("",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def polygon_area(x, y):
    '''
    根据多边形的坐标列表计算面积
    :param x:
    :param y:
    :return:
    '''
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)


