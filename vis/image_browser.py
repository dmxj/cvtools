# -*- coding: utf-8 -*-

#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
"""
This model is provided ...
Author: huangtehui@baidu.com
Date: 2018/XX/XX
"""
import argparse
import cv2
import os
from PIL import Image
import time
import glob
import pygame
import sys
from pynput import keyboard, mouse
import shutil


def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def imgs2video(imgs_dir, save_name):
    fps = 24
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (1280, 960))
    # no glob, need number-index increasing
    imgs = glob.glob(os.path.join(imgs_dir, '*.JPG'))
    for i in range(len(imgs)):
        #imgname = os.path.join(imgs_dir, 'core-{:02d}.png'.format(i))
        imgname = imgs[i]
        print(imgname)
        frame = cv2.imread(imgname)
        video_writer.write(frame)

    video_writer.release()

def imgs_show(rs_dir):
    images = os.listdir(rs_dir)
    pic_num = len(images)
    print("pic_num:{}".format(pic_num))
    pygame.init()
    pass_flag = True

    target_path = os.path.join(rs_dir, images[0])

    height,width = cv2.imread(target_path).shape[:2]
    screen = pygame.display.set_mode((width, height), 0, 32)
    count = 0
    # img = cv2.imread(target_path)
    # cv2.imshow("result", img)
    #img = Image.open(target_path)
    #img.show()
    #str_in = input()
    #keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    #keyboard_listener.start()
    #keyboard_listener.join()
    #print("received input is ", keys)
    img = pygame.image.load(target_path).convert()
    screen.blit(img, (0, 0)) #显示图片，位置（0，0），这个位置是该图片左上角的坐标
    pygame.display.update()
    del_flag = 0
    right_flag = 0
    left_flag = 0

    while pass_flag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        #pygame.event.pump()
        #keys = pygame.key.get_pressed()
            elif event.type == pygame.KEYDOWN:
                #print("event.key: ", event.key)
                if event.key == pygame.K_RIGHT and count <= pic_num:
                    count = count + 1
                    #print("right ")
                    target_path = os.path.join(rs_dir, images[count])
                    img = pygame.image.load(target_path).convert()
                    screen.blit(img, (0, 0))
                    pygame.display.update()

                elif event.key == pygame.K_LEFT and count <= pic_num:
                    count = count - 1
                    target_path = os.path.join(rs_dir, images[count])
                    img = pygame.image.load(target_path).convert()
                    screen.blit(img, (0, 0))
                    pygame.display.update()

                elif event.key == pygame.K_BACKSPACE and count <= pic_num:
                    #print("delete ")
                    dst_path = os.path.join('/Users/huangtehui/.Trash/', images[count])
                    images.pop(count)
                    count = count - 1
                    pic_num = pic_num - 1
                    print("pic_num:{}".format(pic_num))
                    try:
                        if os.path.exists(dst_path):
                            os.remove(dst_path)
                        else:
                            shutil.move(target_path, '/Users/huangtehui/.Trash/')
                            print(target_path, "is moved to .Trash")
                            del_flag = 1
                    except Exception as e:
                        print(e)
                        continue
                    pygame.display.update()




def test_pygame():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))

    done = False
    while not done:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            done = True
        if keys[pygame.K_SPACE]:
            print("got here")
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                print("mouse at ", event.pos)


if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser()
    # 结果文件路径
    parser.add_argument('-p', '--path', default="/Users/rensike/Work/宝钢热轧/baogang_hot_data_v4/hot_test_result_v7/heixian")
    args = parser.parse_args()

    rs_path = args.path
    #imgs2video(rs_path, "/Users/huangtehui/Documents/上海城投/test.avi")
    imgs_show(rs_path)
    #test_pygame()