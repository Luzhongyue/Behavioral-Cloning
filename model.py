import os
import csv
import cv2
import numpy as np
import sklearn 
from keras.models import Sequential,Model
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout,Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.backend import tf 
#import keras.backend.tensorflow_backend as KTF
from sklearn.utils import shuffle
import matplotlib as mping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

samples=[]
data_path='./data_collect/driving_log.csv'
with open (data_path,'r')as f:
    reader=csv.reader(f)
    for row in reader :
        samples.append(row)  
#remove the first row, because the first row are labels.
del(samples[0])


#split the data into training data and validation data
train_samples,validation_samples=train_test_split(samples,test_size=.2)
print('the num of train samples before:',len(train_samples))
print('the num of validation samples before:',len(validation_samples))
#print(samples[0])
#print(type(samples))
#augemented data
#flipping images and taking the opposite sign of the steering measurement
def random_flip(img,steering):
    img1=np.asarray(img)
    coin=np.random.choice([0,1])
    if coin==1:
        img_flipped,steering=np.fliplr(img1),-steering 
    else:
        img,steering=img1,steering
    return img,steering
#randomly adjust brightness of the image
def random_brightness(img,steering):
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    cofficient=np.random.uniform(low=.1,high=1.0,size=None)
    v=hsv[:,:,2]*cofficient
    hsv[:,:,2]=v.astype('uint8')
    bright_img=cv2.cvtColor(hsv.astype('uint8'),cv2.COLOR_HSV2RGB)
    steering=steering
    return bright_img,steering

#define the generator
def generator(samples,batch_size=100):
    num_samples=len(samples)
    while 1:#loop forevor so the generator never terminates
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            car_images=[]
            steering_angles=[]
            for batch_sample in batch_samples:
                center_name='./data_collect/IMG/'+batch_sample[0].split('/')[-1]
                left_name='./data_collect/IMG/'+batch_sample[1].split('/')[-1]
                right_name='./data_collect/IMG/'+batch_sample[2].split('/')[-1]
                steering_center=float(batch_sample[3])
                #create adjusted steering measurement for the side camera images
                correction=0.2
                steering_left=steering_center+correction
                steering_right=steering_center-correction
                #read in images from center,left and right cameras
                #center_image=cv2.imread(center_name)
                center_image=Image.open(center_name).convert('RGB')
                left_image=Image.open(left_name).convert('RGB')
                right_image=Image.open(right_name).convert('RGB')
                #if (center_image == None) or (left_image==None) or (right_image==None): 
                    #raise Exception("could not load image !")
                #center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                #right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                #left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                #add images and angles to data set
                steering_angles.extend((steering_center,steering_left,steering_right))
                car_images.extend((np.array(center_image),np.array(left_image),np.array(right_image)))
                
                #augement data set
                #img_list1=[]
                #img_list2=[]
                #angle_list1=[]
                #angle_list2=[]
                #for image in car_images:
                    #for angle in steering_angles:
                        #image1,angle1=random_flip(image,angle)
                        #image2,angle2=random_brightness(image,angle)
                        #img_list1.append(image1)
                        #img_list2.append(image2)
                        #angle_list1.append(angle1)
                        #angle_list2.append(angle2)
                #car_images.extend(img_list1)
                #car_images.extend(img_list2)
                #steering_angles.extend(angle_list1)
                #steering_angles.extend(angle_list2)
                #print(steering_angles)
                #print(car_images)
                
                x_train=np.array(car_images)
                y_train=np.array(steering_angles)
            yield shuffle(x_train,y_train)
train_generator=generator(train_samples,batch_size=100)
validation_generator=generator(validation_samples,batch_size=100)
print('the num of train samples:',len(train_samples))
print('the num of validation samples:',len(validation_samples))

def resize_img(img):
    from keras.backend import tf 
    img=tf.image.resize_images(img, (66, 200))
    return img

#keras model refered to NVIDIA model
model=Sequential()
#croping the images to remove the noise
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=(160,320,3)))
#normalize and #resize the image to acceralate the training step
model.add(Lambda(resize_img))
model.add(Lambda(lambda x:x/255-0.5))
#add three 5*5 convolution layers,output depth:12,24,36
model.add(Conv2D(12,(5,5),strides=(2,2),padding='valid',activation='relu'))
model.add(Conv2D(24,(5,5),strides=(2,2),padding='valid',activation='relu'))
model.add(Conv2D(36,(5,5),strides=(2,2),padding='valid',activation='relu'))
#add two 3*3 convolution layers,output depth:64,64
model.add(Conv2D(64,(3,3),strides=(1,1),padding='valid',activation='relu'))
model.add(Conv2D(64,(3,3),strides=(1,1),padding='valid',activation='relu'))
#add a flatten layer
model.add(Flatten())
#add three fully connected layers,output depth:100,50,10
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.summary()

#train the model
batch_size=100
model.compile(loss='mse',optimizer='adam')
history_object=model.fit_generator(train_generator,steps_per_epoch=len(train_samples)/batch_size,\
                    validation_data=validation_generator,\
                    validation_steps=len(validation_samples)/batch_size,epochs=5,verbose=1)

#save model
model.save('model_my.h5')
print('model saved')

#Visualize loss
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'],loc='upper right')
plt.show()








                
                
                
                
                
        

    
    
    

        
