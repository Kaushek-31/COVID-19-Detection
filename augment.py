import os
import cv2

def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

myFileList = createFileList(r'C:\Users\Kaushek\Desktop\PRO_2020\Dataset\Traffic\Crop_Dataset_Original')
save_dir = r'C:\Users\Kaushek\Desktop\PRO_2020\Dataset\Traffic\Crop_Dataset_Renamed'

j = 0
for file in myFileList:
    j += 1
    img = cv2.imread(file)
    outpath = f"{j}.jpg"
    cv2.imwrite(outpath, img)
    print(outpath)