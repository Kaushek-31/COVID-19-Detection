import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, add, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D
from keras.layers. normalization import BatchNormalization
import pickle
from keras import optimizers
from keras.optimizers import SGD 
from sklearn.metrics import confusion_matrix
from keras.applications import Xception, ResNet50, VGG16


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

'''#[1,0] for covid, [0,1] for non_covid

Y_train = np.zeros((y_train.size, 2))
for i in range(y_train.size):
  Y_train[i][int(1-y_train[i])] = 1
'''

#Developing the CNN model

print("Devoloping a model")
batch_size = 16
num_classes = 1
epochs = 100
img_size = 100

'''def residual(curr_layer, filters_no):
    m_layer = curr_layer
    if curr_layer.shape[-1] != filters_no:
        m_layer = Conv2D(filters_no, (1,1), padding = 'same')(curr_layer)
    
    l1 = Conv2D(filters_no, (3,3), padding='same', activation='relu')(curr_layer)
    l2 = BatchNormalization()(l1)
    
    l3 = Conv2D(filters_no, (3,3), padding='same', activation='tanh')(l2)
    l4 = BatchNormalization()(l3)
    
    add_layer = add([m_layer, l4])
    out_layer = Activation('relu')(add_layer)
    return out_layer
    
start = Input(shape = (img_size, img_size, 3))
c_1 = Conv2D(64, (3, 3), padding = 'same', activation='relu')(start)
c_2 = MaxPooling2D(pool_size=(2, 2))(c_1)
res_1 = residual(c_2, 64)
res_3 = residual(res_1, 128)
res_4 = residual(res_3, 256)
res_5 = residual(res_4, 512)

max_p = MaxPooling2D(pool_size=(3,3))(res_5)
flat = Flatten()(max_p)
dense2 = Dense(64, activation='relu')(flat)
output = Dense(num_classes, activation = 'sigmoid')(dense2)

model = Model(inputs = start, outputs = output)'''

model = Sequential()
model.add(ResNet50(weights = 'imagenet', include_top=False, input_shape=(img_size, img_size, 3)))
model.add(Flatten())
model.add(Dense(num_classes, activation = 'sigmoid'))

opt = opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

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

filename = r'C:\Users\Kaushek\Desktop\PRO_2020\Test Results\Pickle files\Test.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

#predicting the model

prediction = model.predict(X_test, verbose = 1)

'''for i in range(y_test.size):
    result = prediction[i][0] > prediction[i][1] 
    if result == 1:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
'''
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


