import numpy as np
from pandas import read_csv
import os
import csv
import cv2
import matplotlib.pyplot as plt

#Useful function
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=True):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


def resize(img):
    n_height = 75
    n_width = 75
    re_img = cv2.resize(img, (n_width, n_height), interpolation = cv2.INTER_AREA)
    return re_img



fileList = r'C:\Users\kkram\Desktop\PRO_2020\Dataset\Classification\cogan_noncovid'    
csv_file = r'C:\Users\kkram\Desktop\PRO_2020\Codes\noncovid_cogan_75.csv'    
myFileList = createFileList(fileList) 
    
i = 0
for file in myFileList:
    print(i)
    i += 1
    img_file = cv2.imread(file)
    img = resize(img_file)
    value = np.asarray(img, dtype=np.int)
    value = value.flatten()
    with open(csv_file, mode='a') as f:
        writer = csv.writer(f)
        writer.writerow(value)
            

    