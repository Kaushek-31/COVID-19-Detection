import numpy as np
from pandas import read_csv
import os
import csv
import cv2
import matplotlib.pyplot as plt

#Useful function
def createFileList(myDir, format='.ppm'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

def createcsvList(myDir, format='.csv'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


def createDirList(myDir):
    fileList = []
    print(myDir)
    for root, dirs, file in os.walk(myDir, topdown=False):
        for name in dirs:
            fullName = os.path.join(root, name)
            fileList.append(fullName)
    return fileList

def resize(img):
    n_height = 32
    n_width = 32
    re_img = cv2.resize(img, (n_width, n_height), interpolation = cv2.INTER_AREA)
    return re_img



save_x = r'C:\Users\Kaushek\Desktop\PRO_2020\Dataset\Traffic\CSV\train_traffic_x.csv'
save_y = r'C:\Users\Kaushek\Desktop\PRO_2020\Dataset\Traffic\CSV\train_traffic_y.csv'
directory = r'C:\Users\Kaushek\Desktop\PRO_2020\Dataset\Traffic\Train\GTSRB\Final_Training\Images'

d = createDirList(directory)

for directory in d:
    myFileList = createFileList(directory)
    csvlist = createcsvList(directory)
    trainy = read_csv(csvlist[0])
    
    i = 0
    for file in myFileList:
        print(file)
        img_file = cv2.imread(file)
        img = resize(img_file)
        value = np.asarray(img, dtype=np.int)
        value = value.flatten()
        with open(save_x, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)
        y_data = trainy['Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassIdS'][i].split(';')
        y_data.pop(0)
        with open(save_y, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(y_data)
        i += 1    