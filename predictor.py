import pickle
import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix as confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

filename = r'C:\Users\Kaushek\Desktop\PRO_2020\Test Results\Pickle files\Traffic\traffic.h5'

model = load_model(filename)

test_set_x = pd.read_csv(r'C:\Users\Kaushek\Desktop\PRO_2020\Dataset\Traffic\CSV\train_traffic_x.csv', header=None)
test_set_y = pd.read_csv(r'C:\Users\Kaushek\Desktop\PRO_2020\Dataset\Traffic\CSV\train_traffic_y.csv', header=None) 

y_list = ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60',
          'speed limit 70', 'speed limit 80' ,'restriction ends',
          'speed limit 100', 'speed limit 120', 'no overtaking', 'no overtaking',
          'priority at next intersection', 'priority road', 'give way', 'stop',
          'no traffic both ways', 'no trucks', 'no entry', 'danger', 'bend left',
          'bend right', 'bend', 'uneven road', 'slippery road', 'road narrows',
          'construction', 'traffic signal', 'pedestrian crossing', 'school crossing',
          'cycles crossing', 'snow', 'animals', 'restriction ends', 'go right', 'go left',
          'go straight', 'go right or straight', 'go left or straight', 'keep right',
          'keep left', 'roundabout', 'restriction ends', 'restriction ends(truck)']

X_test = test_set_x.iloc[:,:].values
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
y_test = test_set_y.iloc[:,-1].values
y_test = y_test.reshape(y_test.shape[0], 1)
Y_test = np.zeros((y_test.shape[0], 43))
for i in range(y_test.shape[0]):
    Y_test[i][y_test[i][0]] = 1 
prediction = model.predict(X_test, verbose = 1)

preds = np.zeros((prediction.shape[0], 43))
for i in range(prediction.shape[0]):
    maxi = 0
    ind = 0
    for j in range(0, 43, 1):
        if prediction[i][j] > maxi:
            preds[i][ind] = 0
            ind = j
            maxi = prediction[i][j]
            preds[i][j] = 1

pred = np.zeros((prediction.shape[0], 1))
for i in range(prediction.shape[0]):
    for j in range(0, 43, 1):
        if preds[i][j] == 1:
            pred[i] = j
            break
c_matrix = confusion_matrix(y_test, pred)

print("The confusion matrix is:")
print(c_matrix)
print('')

acc = accuracy_score(y_test, pred)
prec = precision_score(Y_test, preds, average=None)
recall = recall_score(Y_test, preds, average=None)
f1 = f1_score(Y_test, preds, average=None)

print(f">>>>> The overall accuracy of the model is: {acc}% <<<<<")
print(f">>>>> Precision is: {prec} <<<<<")
print(f">>>>> Recall is: {recall} <<<<<") 
print(f">>>>> F1 score is: {f1} <<<<<")
#Dumping the model