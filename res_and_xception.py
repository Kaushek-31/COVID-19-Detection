import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Add, Input, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D
from keras.layers. normalization import BatchNormalization
import pickle
from keras import optimizers
from keras.optimizers import SGD 
from sklearn.metrics import confusion_matrix
from keras.applications import Xception, ResNet50, VGG16
import math

#Importing the dataset

covid = pd.read_csv(r'C:\Users\Kaushek\Desktop\PRO_2020\Codes\covid_2500.csv', header=None)
non_covid = pd.read_csv(r'C:\Users\Kaushek\Desktop\PRO_2020\Codes\non_covid_2500.csv', header=None)

#Adding the dependent variable

result = [1]*covid.shape[0]
covid['result'] = result
result = [0]*non_covid.shape[0]
non_covid['result'] = result

dataset= pd.concat([covid,non_covid])
X= dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(dataset.shape)

#Splitting the dataset

X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.25, random_state = 42)

#Shaping X data and converting Y data to a one hot vector

X_train = X_train.reshape(X_train.shape[0], 100, 100, 3)
X_test = X_test.reshape(X_test.shape[0], 100, 100, 3)

#Developing the CNN model

print("Devoloping a model")
batch_size = 16
num_classes = 1
epochs = 100
batch_per_epoch = 50
img_size = 100

def addition_lay(curr_layer, filter_no1, filter_no2):
    
    l1 = Conv2D(filter_no1, kernel_size=(3,3), padding='same', activation='relu')(curr_layer)
    a1 = BatchNormalization()(l1)
    l2 = Conv2D(filter_no2, kernel_size=(3,3), padding='same', activation='linear')(curr_layer)
    l3 = Conv2D(filter_no1, (1,1), padding='same', activation='relu')(l2)
    a2 = BatchNormalization()(l3)
    
    add_layer = Add()([a1, a2])
    pool_layer = AveragePooling2D(pool_size=(2,2), padding='same')(add_layer)
    norm_layer = BatchNormalization()(pool_layer)
    out_layer = Activation('relu')(norm_layer)
    return out_layer

start = Input(shape = (img_size, img_size, 3))
l1 = Conv2D(64, kernel_size=(3, 3), activation='relu')(start)
l2 = MaxPooling2D(pool_size=(2,2), strides=(1,1))(l1)
l3 = BatchNormalization()(l2)
l4 = Conv2D(128, kernel_size=(3, 3), activation='relu')(l3)
l5 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l4)
l6 = BatchNormalization()(l5)
l7 = Conv2D(256, kernel_size=(3, 3), activation='relu')(l6)
l8 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l7)
l9 = BatchNormalization()(l8)
l10 = Conv2D(512, kernel_size=(3, 3), activation='relu')(l9)
l11 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l10)
l12 = BatchNormalization()(l11)
l13 = Flatten()(l12)
l14 = Dense(512, activation='relu')(l13)
l15 = Dropout(0.3)(l14)
l16 = Dense(64, activation='relu')(l15)
l17 = Dropout(0.2)(l16)
end = Dense(num_classes, activation='sigmoid')(l17)

model = Model(inputs=start, outputs=end)

opt = optimizers.RMSprop(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])
model.summary()

print("Compilation done")

#Training
trainingsize = 5000 
validate_size = 1000

def calculate_spe(y):
  return int(math.ceil((1. * y) / batch_size)) 

steps_per_epoch = calculate_spe(trainingsize)
validation_steps = calculate_spe(validate_size)

history = model.fit(X_train, y_train, epochs=epochs,
                    verbose=1, validation_split=0.2, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=validation_steps)

print("................TRAINING DONE....................")

#plotting the accuracy and loss

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Dumping the model

filename = r'C:\Users\Kaushek\Desktop\PRO_2020\Test Results\Pickle files\Test.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

#predicting the model

prediction = model.predict(X_test, verbose = 1)

threshold = 0.7
y_pred = [0]*prediction.size

for i in range(prediction.size):
    y_pred[i] = prediction[i] > threshold
    
c_matrix = confusion_matrix(y_test, y_pred)

print("The confusion matrix is:")
print(c_matrix)
print('')

tp = c_matrix[0][0]
fn = c_matrix[0][1]
fp = c_matrix[1][0]
tn = c_matrix[1][1]

acc = round((tp+tn)/(tp+tn+fp+fn) * 100, 2)

prec = tp/(tp+fp)
recall = tp/(tp+fn)
sensitivity = tp / float(fn + tp)
specificity = tn / float(fp + tn)
f1 = round((2*prec*recall)/(prec+recall) * 100, 2)

print(f">>>>> The overall accuracy of the model is: {acc}% <<<<<")
print(f">>>>> Precision is: {prec} <<<<<")
print(f">>>>> Recall is: {recall} <<<<<")
print(f">>>>> Sensitivity is: {sensitivity} <<<<<")
print(f">>>>> Specificity is: {specificity} <<<<<")
print(f">>>>> F1 score is: {f1} <<<<<")  