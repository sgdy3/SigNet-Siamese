# -*- coding: utf-8 -*-
# ---
# @File: texture_mat.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/22
# Describe: 实现了signet-f
# ---



import cv2 as cv
from keras.layers import Conv2D,MaxPool2D,BatchNormalization,Dense,GlobalAveragePooling2D,Lambda
from keras.models import Sequential,Model
from keras.layers import Activation,Dropout,Input,Flatten
from keras.optimizer_v2.rmsprop import RMSprop
from keras import backend as K
from keras.utils.vis_utils import plot_model
import numpy as np
import tensorflow as tf
import os


class Siamase():
    def __init__(self,mod='thin'):
        self.rows=150
        self.cols=220
        self.channles=1
        self.imgshape = (self.rows, self.cols, self.channles)

        self.batchsize=32
        self.epochs=1

        assert mod=='thin' or 'std' ,"model has only two variant: thin and std"

        self.subnet=self.model_thin() if mod=='thin' else self.model_std()
        self.optimizer= RMSprop(learning_rate=1e-4,rho=0.9,epsilon=1e-8,decay=5e-4)

        sig1=Input(shape=self.imgshape)
        sig2=Input(shape=self.imgshape)
        label=Input(shape=(1,)) # denote pairs

        feature1=self.subnet(sig1)
        feature2=self.subnet(sig2)

        dw=Lambda(self.Eucilid,name='distance')([feature1,feature2])
        contra_loss=Lambda(self.build_loss,name='loss')([dw,label])
        self.SigNet=Model([sig1,sig2,label],[dw,contra_loss])

        loss_layer=self.SigNet.get_layer('loss').output
        self.SigNet.add_loss(loss_layer)
        self.SigNet.compile(optimizer=self.optimizer)
        plot_model(self.SigNet, to_file='siamese.png', show_shapes=True)
        self.SigNet.summary()

    def model_thin(self):
        model=Sequential()

        model.add(Conv2D(48,kernel_size=(11,11),strides=1,padding='same',input_shape=(self.rows,self.cols,self.channles),name='conv1',activation='relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),name='pool1'))

        model.add(Conv2D(64, kernel_size=(5, 5), strides=1, padding='same',name='conv2',activation='relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(MaxPool2D(pool_size=(3, 3), strides=2,name='pool2'))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding='same',name='conv3',activation='relu'))

        model.add(Conv2D(96, kernel_size=(3, 3), strides=1, padding='same',name='conv4',activation='relu'))


        model.add(MaxPool2D(pool_size=(3, 3), strides=2,name='pool3'))
        model.add(Dropout(0.3))


        model.add(GlobalAveragePooling2D())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128,name='fc2',activation='relu'))

        model.summary()

        img=Input(shape=self.imgshape)
        feature=model(img)
        plot_model(model,to_file='subnet.png',show_shapes=True)
        return  Model(img,feature)

    def model_std(self):
        model=Sequential()

        model.add(Conv2D(96,kernel_size=(11,11),strides=1,padding='same',input_shape=(self.rows,self.cols,self.channles),name='conv1',activation='relu'))
        model.add(BatchNormalization(epsilon=1e-06,momentum=0.9))

        model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),name='pool1'))

        model.add(Conv2D(256, kernel_size=(5, 5), strides=1, padding='same',name='conv2',activation='relu'))
        model.add(BatchNormalization(epsilon=1e-06,momentum=0.9))

        model.add(MaxPool2D(pool_size=(3, 3), strides=2,name='pool2'))
        model.add(Dropout(0.3))

        model.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding='same',name='conv3',activation='relu'))

        model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding='same',name='conv4',activation='relu'))

        model.add(MaxPool2D(pool_size=(3, 3), strides=2,name='pool3'))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1024,name='fc1',activation='relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128,name='fc2',activation='relu'))
        model.summary()
        img=Input(shape=self.imgshape)
        feature=model(img)
        plot_model(model,to_file='subnet.png',show_shapes=True)
        return  Model(img,feature)

    def  Eucilid(self,args):
        feature1,feature2= args
        dw = K.sqrt(K.sum(K.square(feature1 - feature2),axis=1))  # Euclidean distance
        return  dw

    def build_loss(self,args,alpha=0.5,beta=0.5):
        dw,label=args
        hingeloss=K.maximum(1-dw,0) # maximum(1 - y_true * y_pred, 0)
        label=tf.cast(label,tf.float32) # previously setting label int8 type tensor,can't do element wise with dw
        contrastive_loss=K.sum(label*alpha*K.square(dw)+(1-label)*beta*K.square(hingeloss))/self.batchsize
        return contrastive_loss

    def train(self,dataset,weights='',save=False):
        save_dir = './NetWeights/Siamase_weights'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if weights:
            filepath = os.path.join(save_dir, weights)
            self.SigNet.load_weights(filepath)
        else:
            filepath = os.path.join(save_dir, 'siamase_weight.h5')
            dataset = dataset.shuffle(100).batch(self.batchsize).repeat(self.epochs)
            i=0
            doc=[]
            pre_loss=[]
            min_loss=100 # manually set
            for batch in dataset:
                loss = self.SigNet.train_on_batch(batch)
                doc.append(loss)
                print("%d round loss: %f " % (i, loss))
                if(early_stop(20,loss,pre_loss,threshold=0.5)):
                    print("training complete")
                    break
                if(i>500):
                    print("enough rounds!!")
                    break
                i+=1
            if save:
                self.SigNet.save_weights(filepath)
            return doc

def early_stop(stop_round,loss,pre_loss,threshold=0.005):
    '''
    early stop setting
    :param stop_round: rounds under caculated
    :param pre_loss: loss list
    :param threshold: minimum one-order value of loss list
    :return: whether or not to jump out
    '''
    if(len(pre_loss)<stop_round):
        pre_loss.append(loss)
        return False
    else:
        loss_diff=np.diff(pre_loss,1)
        pre_loss.pop(0)
        pre_loss.append(loss)
        if(abs(loss_diff).mean()<threshold): # to low variance means flatten field
            return True
        else:
            return False