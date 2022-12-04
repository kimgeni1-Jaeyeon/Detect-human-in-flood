import os
import zipfile
import random
import shutil
import tensorflow as tf
from keras.optimizers import RMSprop ##
from keras.preprocessing.image import ImageDataGenerator ##
from shutil import copyfile
from os import getcwd
from os import listdir
import cv2
from tensorflow.python.keras.layers import Conv2D, Input, ZeroPadding2D, Activation, MaxPooling2D, Flatten, Dense ##BatchNormalization
from tensorflow.python.keras.models import Model, load_model ##
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint ##
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    dataset = []
    
    for dataimage in os.listdir(SOURCE):
        data = SOURCE + dataimage
        if(os.path.getsize(data) > 0):
            dataset.append(dataimage)
        else:
            print('Skipped ' + dataimage)
            print('There is no file.')
    
    train_len = int(len(dataset) * SPLIT_SIZE)
    test_len = int(len(dataset) - train_len)
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = dataset[0:train_len]
    test_set = dataset[-test_len:]
       
    for dataimage in train_set:
        temp_train_set = SOURCE + dataimage
        final_train_set = TRAINING + dataimage
        copyfile(temp_train_set, final_train_set)
    
    for dataimage in test_set:
        temp_test_set = SOURCE + dataimage
        final_test_set = TESTING + dataimage
        copyfile(temp_test_set, final_test_set)
        
        
YES_SOURCE_DIR = "Faceimage/test/augmented data/yesreal/"
TRAINING_YES_DIR = "Faceimage/test/augmented data/training/yes/"
TESTING_YES_DIR = "Faceimage/test/augmented data/testing/yes/"
NO_SOURCE_DIR = "Faceimage/test/augmented data/noreal/"
TRAINING_NO_DIR = "Faceimage/test/augmented data/training/no/"
TESTING_NO_DIR = "Faceimage/test/augmented data/testing/no/"
YES_SOURCE_DIR2 = "Bodyimage/test/augmented data/yesreal/"
TRAINING_YES_DIR2 = "Bodyimage/test/augmented data/training/yes/"
TESTING_YES_DIR2 = "Bodyimage/test/augmented data/testing/yes/"
NO_SOURCE_DIR2 = "Bodyimage/test/augmented data/noreal/"
TRAINING_NO_DIR2 = "Bodyimage/test/augmented data/training/no/"
TESTING_NO_DIR2 = "Bodyimage/test/augmented data/testing/no/"
split_size = .8
split_data(YES_SOURCE_DIR, TRAINING_YES_DIR, TESTING_YES_DIR, split_size)
split_data(NO_SOURCE_DIR, TRAINING_NO_DIR, TESTING_NO_DIR, split_size)
split_data(YES_SOURCE_DIR2, TRAINING_YES_DIR2, TESTING_YES_DIR2, split_size)
split_data(NO_SOURCE_DIR2, TRAINING_NO_DIR2, TESTING_NO_DIR2, split_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(100, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(100, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#model = load_model('face_model.h5')
#model2 = load_model('body_model.h5')

#학습된 모델을 load_model로 불러올 시 건너뛰세요

TRAINING_DIR = "Faceimage/test/augmented data/training"
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                    batch_size=10, 
                                                    target_size=(150, 150))
VALIDATION_DIR = "Faceimage/test/augmented data/testing"
validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                         batch_size=10, 
                                                         target_size=(150, 150))
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

#######

TRAINING_DIR2 = "Bodyimage/test/augmented data/training"
train_datagen2 = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator2 = train_datagen2.flow_from_directory(TRAINING_DIR2, 
                                                    batch_size=10, 
                                                    target_size=(150, 150))
VALIDATION_DIR2 = "Bodyimage/test/augmented data/testing"
validation_datagen2 = ImageDataGenerator(rescale=1.0/255)

validation_generator2 = validation_datagen2.flow_from_directory(VALIDATION_DIR2, 
                                                         batch_size=10, 
                                                         target_size=(150, 150))
checkpoint2 = ModelCheckpoint('model2-{epoch:03d}.model2',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

facelearning = model.fit(train_generator,
                              epochs=1,
                              validation_data=validation_generator,
                              callbacks=[checkpoint])

bodylearning = model2.fit(train_generator2,
                              epochs=1,
                              validation_data=validation_generator2,
                              callbacks=[checkpoint2])

model.save('face_model.h5')
model2.save('body_model.h5')

labels_dict={0:'',1:'Face'}
labels2_dict={0:'',1:'Body'}
color_dict={0:(0,255,0),1:(0,255,0)}
color2_dict={0:(0,0,255),1:(0,0,255)}

size = 4
REC = cv2.VideoCapture('Lagos flood today.mp4') #영상 변경하는 곳

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier2 = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    (rval, im) = REC.read()

    # 읽어들이는 이미지, 영상 크기 조정
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    faces = classifier.detectMultiScale(mini)
    bodies = classifier2.detectMultiScale(mini)

    # 포착된 사람에 테두리 표시
    for f in faces:
        (x, y, w, h) = [v * size for v in f] 
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        print(result)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    for f in bodies:
        (x, y, w, h) = [v * size for v in f]
        body_img = im[y:y+h, x:x+w]
        resized=cv2.resize(body_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model2.predict(reshaped)
        #print(result)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),color2_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color2_dict[label],-1)
        cv2.putText(im, labels2_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        

    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    if key == 27:
        break

REC.release()
cv2.destroyAllWindows()
