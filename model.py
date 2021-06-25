import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers. normalization import BatchNormalization
import pickle
from keras import optimizers as optimizers
from sklearn.metrics import confusion_matrix

#Importing the dataset

covid = pd.read_csv(r'C:\Users\Kaushek\Desktop\PRO_2020\Codes\covid_2500.csv', header=None)
non_covid = pd.read_csv(r'C:\Users\Kaushek\Desktop\PRO_2020\Codes\non_covid_2500.csv', header=None)
print("Done reading .csv files")

#Adding the dependent variable
result = [1]*covid.shape[0]
covid['result'] = result
result = [0]*non_covid.shape[0]
non_covid['result'] = result
print('Done resulting')

dataset= pd.concat([covid,non_covid])
X= dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(dataset.shape)

#Splitting the dataset
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.25, random_state = 42)

X_train = X_train.reshape(X_train.shape[0], 64, 64, 3)
X_test = X_test.reshape(X_test.shape[0], 64, 64, 3)

plt.imshow(X_train[123])
plt.show()

#Developing the CNN model
print("Devoloping a model")
batch_size = 16
num_classes = 1
epochs = 50
img_size = 64

model = Sequential()
model.add(Conv2D(128, kernel_size = (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size = (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(512, kernel_size = (3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(1024, kernel_size= (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(512, kernel_size= (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation = 'sigmoid'))

#Compilation
opt = optimizers.Adam(learning_rate=0.01, amsgrad=True)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])

model.summary()
print("Compilation done")

#Training
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)
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

filename = r'C:\Users\Kaushek\Desktop\PRO_2020\Test Results\Pickle files\Test - final.pkl'
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