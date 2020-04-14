"""
    CMPT 414 Project
    author: Yiming Zhang, Chongyu Yuan
"""

from PIL import Image
import os
import random

valid_root = "dataset/valid_images/"
train_root = "dataset/train_images/"


class Data_loader:
    def __init__(self):
        self.train_images = []
        self.valid_images = []
        for i in range(34):
            train_dir = train_root + str(i)
            valid_dir = valid_root + str(i)
            files = os.listdir(train_dir)
            for file in files:
                self.train_images.append((convert_img_to_list(train_dir + '/' + file), i))
            files = os.listdir(valid_dir)
            for file in files:
                self.valid_images.append((convert_img_to_list(valid_dir + '/' + file), i))
        random.shuffle(self.train_images)
        random.shuffle(self.valid_images)
        self.train_count = len(self.train_images)
        self.valid_count = len(self.valid_images)
        self.curr_index = 0

    def load_random_data(self, size, is_validation=False):
        imgs = []
        labels = []
        if is_validation:
            index = self.curr_index % self.valid_count
            index2 = (index + size) % self.valid_count
            if index2 <= index:
                index2 = self.valid_count
            for i in range(index, index2):
                tmp = [0] * 34
                tmp[self.valid_images[i][1]] = 1
                imgs.append(self.valid_images[i][0])
                labels.append(tmp)
        else:
            index = self.curr_index % self.train_count
            index2 = (index + size) % self.train_count
            if index2 <= index:
                index2 = self.train_count
            for i in range(index, index2):
                tmp = [0] * 34
                tmp[self.train_images[i][1]] = 1
                imgs.append(self.train_images[i][0])
                labels.append(tmp)
        self.curr_index += size
        return imgs, labels


def convert_img_to_list(image_name):
    image = Image.open(image_name)
    return [[image.getpixel((x, y)) for x in range(image.size[0])] for y in range(image.size[1])]
