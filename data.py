import cv2 as cv
import os
import numpy as np


data_path = "dataset"
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))

IMG_SIZE = 96


def data_preprocessing(size: int = IMG_SIZE):
    x = []
    y = []
    for category in categories:
        folder_path = os.path.join(data_path, category)
        img_names = os.listdir(folder_path)

        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            resized = cv.resize(img, (size, size))

            x.append(resized)
            y.append(label_dict[category])
    
    x = np.array(x) / 255
    y = np.array(y)
    x = x.reshape(x.shape[0], size, size, 3)

    return x, y
