# -*- coding: utf-8 -*-
# ---
# @File: main.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/6/5
# Describe: 
# ---


import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from model import Siamase
from preprocessing import hafemann_preprocess

def load_img(file_name1,file_name2,label,ext_h=820,ext_w=890):
    # CEDAR (829,890)
    # sigcomp (1100,2900)

    img1 = tf.io.read_file(file_name1, 'rb')  # 读取图片
    img1 = tf.image.decode_png(img1, channels=3)
    img1 = tf.image.rgb_to_grayscale(img1)
    img1=hafemann_preprocess(img1.numpy(),ext_h,ext_w)
    img1=np.expand_dims(img1,-1)
    img2 = tf.io.read_file(file_name2, 'rb')  # 读取图片
    img2 = tf.image.decode_png(img2, channels=3)
    img2 = tf.image.rgb_to_grayscale(img2)
    img2=hafemann_preprocess(img2.numpy(),ext_h,ext_w)
    img2=np.expand_dims(img2,-1)
    return img1,img2,label

def judegement(T,N,d):
    '''

    :param TP: (ndarry) pos_pairs' distance
    :param NP: (ndarry) neg_pairs' distance
    :param d: (float32)
    :return: accuarcy
    '''
    TPR=(T[:,0]<d).sum()/T.shape[0]
    TNR=(N[:,0]>d).sum()/T.shape[0]
    acc=1/2*(TPR+TNR)
    return acc


def curve_eval(label,result):
    fpr, tpr, thresholds = roc_curve(label,result, pos_label=0)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=result.copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    pred_label=1-pred_label
    acc=(pred_label==label).sum()/label.size
    area = auc(fpr, tpr)
    print("EER:%f"%EER)
    print('AUC:%f'%area)
    print('ACC(EER_threshold):%f'%acc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on testing set')
    plt.legend(loc="lower right")
    plt.show()


if __name__=='__main__':
    SigNet=Siamase()
    with open('./pair_ind/cedar_ind/train_index.pkl', 'rb') as train_index_file:
        train_ind = pickle.load(train_index_file)
    train_ind = np.array(train_ind)
    # pre-shuffle the training-set
    train_ind=train_ind[np.random.permutation(train_ind.shape[0]),:]

    dataset = tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1], train_ind[:, 2].astype(np.int8)))

    image = dataset.map(
        lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8, tf.uint8, tf.int8]))
    doc=SigNet.train(image,save=True)

    mode='train'

    assert  mode=='train' or mode== 'test', 'the programmer can only execute in training or testing model'

    if mode=='train':
        if(doc): # 进行了训练就画图，直接读档模型不画图
            plt.plot(doc)
            plt.title('contrastive_loss curve')
            plt.xlabel('times')
            plt.ylabel('contrastive loss')

        '''
        给出训练集上所有图片的输出，并进行合并
        '''
        result=[]
        label=[]
        cost=[]
        for b in image.batch(32):
            result.append(SigNet.SigNet.predict_on_batch(b)[0])
            cost.append(SigNet.SigNet.predict_on_batch(b)[1])
            label.append(b[2].numpy())

        # 由于用于数据集大小不一定能整除batch，结果shape不同，不能直接合并
        temp=np.zeros((1,2))
        for i in result:
            temp=np.vstack([temp,i])
        temp=temp[1:,:]
        result=temp.copy()
        temp=np.array([])
        for i in label:
            temp=np.concatenate([temp,i])
        label=temp.copy()
        cost=np.array(cost).reshape(-1,1)
        curve_eval(label,result)
        temp_result=np.vstack([result,label])

        '''
        根据训练集上结果计算判定阈值
        '''
        # ensure threshold according to lecture
        T=temp_result[temp_result[:,1]==1,:]
        N=temp_result[temp_result[:,1]==0,:]
        print(N.mean(axis=0))
        print(T.mean(axis=0))
        d=np.arange(0,1,0.01)
        acc=[]
        for i in d:
            acc.append(judegement(T,N,i))
        acc=np.array(acc)
        threshold=d[acc.argmax()]

    else:
        threshold=0 # must implement training stage to specify threshold
        with open('./pair_ind/cedar_ind/test_index.pkl', 'rb') as test_index_file:
            test_ind = pickle.load(test_index_file)
        test_ind = np.array(test_ind)
        test_set= tf.data.Dataset.from_tensor_slices((test_ind[:, 0], test_ind[:, 1], test_ind[:, 2].astype(np.int8)))
        test_image = test_set.map(
            lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8, tf.uint8, tf.int8]))

        '''
        给出测试集上所有图片的输出，并进行合并
        '''
        result=[]
        label=[]
        cost=[]
        for b in test_image.batch(32):
            result.append(SigNet.SigNet.predict_on_batch(b)[0])
            cost.append(SigNet.SigNet.predict_on_batch(b)[1])
            label.append(b[2].numpy())
        temp=np.array([])
        for i in result:
            temp=np.concatenate([temp,i])
        result=temp.copy()
        temp=np.array([])
        for i in label:
            temp=np.concatenate([temp,i])
        label=temp.copy()
        cost=np.array(cost).reshape(-1,1)  # 画ROC曲线
        curve_eval(label,result)

        # 计算准确率
        temp_result=np.vstack([result,label])
        T=temp_result[temp_result[:,1]==1,:]
        N=temp_result[temp_result[:,1]==0,:]
        print(N.mean(axis=0))
        print(T.mean(axis=0))
        acc=judegement(T,N,threshold)

def USVM():
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'

    final_result=[]
    for user in range(1,16): # 测试15个用户吧
        train_data=[]
        # 15个正样本，15个random forgies做负样本训练SVM
        for j in range(1,16): # 这里不用随机应该没啥问题，让代码好看点吧
            train_data.append([org_path%(user,j),1])
        for j in range(user,user+15):  # 这里不用随机应该没啥问题，让代码好看点吧
            train_data.append([org_path%(j,2),0])
        train_imgs=[]
        for i in train_data:
            train_img = tf.io.read_file(i[0], 'rb')  # 读取图片
            train_img = tf.image.decode_png(train_img, channels=3)
            train_img = tf.image.rgb_to_grayscale(train_img)
            train_img=preprocess(train_img,820,980)
            train_imgs.append([train_img])

        train_imgs=tf.data.Dataset.from_tensor_slices(train_imgs)
        train_vecs=SigNet.subnet.predict(train_imgs)
        label=np.concatenate([np.ones(15),np.zeros(15)])
        label=label.astype(np.int32)  # opencv里SVM要求label需要为int32类型
        svm=cv.ml.SVM_create()
        svm.setKernel(cv.ml.SVM_RBF)
        svm.setType(cv.ml.SVM_C_SVC)
        result=svm.train(train_vecs,cv.ml.ROW_SAMPLE,label)

        test_data=[]
        label=np.zeros(27)
        label[0:9]=1
        # 测试时使用9个正样本，18个负样本（9个random forgies，9个skilled forgies）
        for j in range(16,25):
            test_data.append([org_path%(user,j),1])
        for j in range(user+16,user+25):
            test_data.append([org_path%(j,2),0])
        for j in range(1,10):
            test_data.append([forg_path%(user,j),0])

        test_imgs=[]
        for i in test_data:
            test_img = tf.io.read_file(i[0], 'rb')  # 读取图片
            test_img = tf.image.decode_png(test_img, channels=3)
            test_img = tf.image.rgb_to_grayscale(test_img)
            test_img=preprocess(test_img,820,980)
            test_imgs.append([test_img])
        test_imgs=tf.data.Dataset.from_tensor_slices(test_imgs)
        test_vecs=SigNet.subnet.predict(test_imgs)
        result=svm.predict(test_vecs)[1]
        temp=np.hstack([result,label.reshape(-1,1)])
        final_result.append(temp)