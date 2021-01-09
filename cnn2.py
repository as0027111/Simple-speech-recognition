
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import cv2
# from sklearn.model_selection import train_test_split
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint ,EarlyStopping
from keras.optimizers import SGD, Adam
import os


# In[3]:


# 圖片資料夾路徑
path = './dataset_spectrogram/'


# In[9]:


#把有中文的檔名全部換為數字編號，否則resize時會出錯
import os


f = os.listdir(path)#所有種類列表
# print(f)
# print(len(f))
for dirs in f:
    
    pathh=path+dirs+'/'
#     print(pathh)
    file = os.listdir(pathh)#單一種類當中的所有圖片檔名
#     print(f[0])
    n = 0
    i = 0
#     print(file)
    if(file[0]!='1.jpg'):#若此資料夾內還沒改為英數檔名
        
        for i in file:
#             print(i)
            # 設定舊檔名（就是路徑+檔名）
            oldname = file[n]

            # 設定新檔名
            newname = str(n+1) + '.jpg'
            
            # 用os模組中的rename方法對檔案改名
            os.rename(pathh+oldname, pathh+newname)
            print(oldname, '======>', newname)

            n += 1


# In[ ]:


#資料夾名稱轉英文，否則resize時會出錯
# target=['n','y','不行','好'] #=f
# en_target = ['time', 'unknown', 'weather']
# for i in range(len(f)):
#     os.rename(path+f[i],path+en_target[i])
#     print(f[i], '======>', en_target[i])


# In[3]:


#讀取圖檔並resize
#回傳圖片路徑、圖片(data)、圖片種類(label)
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    fpath = []
    for idx, folder in enumerate(cate):
        # 遍历整个目录判断每个文件是不是符合
        for im in glob.glob(folder + '/*.jpg'):
            #print('reading the images:%s' % (im))
            img = cv2.imread(im)               #opencv讀圖片
            img = cv2.resize(img, (150, 150))  #resize
            imgs.append(img)                 #資料
            labels.append(idx)               #種類別
#            fpath.append(path+im)            #图像路径名
            fpath.append(im)            #图像路径名
            #print(path+im, idx)
            
    return np.asarray(fpath, np.string_), np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


# In[4]:


# 讀取圖檔
fpaths, data, label = read_img(path)
print(data.shape)  # (1000, 256, 256, 3)
# 計算共分為幾類
num_classes = len(set(label))
print(num_classes)


# In[165]:


# # 資料切分train & validation
# total_x = np.array(data)
# total_y = np.array(keras.utils.to_categorical(label, 2)) # 針對Label進行One hot encoding
        
# # 將資料切分為訓練資料集與驗證資料集
# X_train, X_valid, y_train, y_valid = train_test_split(total_x, total_y, train_size=0.8 ,shuffle = 'true') 
# X_train = X_train.astype('float32') / 255.
# X_valid = X_valid.astype('float32') / 255.

# print("Train", X_train.shape, y_train.shape)
# print("Test", X_valid.shape, y_valid.shape)


# In[5]:


#將資料轉為可以進入cnn的形式並生成訓練驗證測試集

print(label)
label = np.array(keras.utils.to_categorical(label, num_classes))#將label轉為二進制的one_hot encoding
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)# 生成等差数列隨機調整圖片順序
data = data[arr]
label = label[arr]
fpaths = fpaths[arr]

# 分割訓練、測試、驗證集 (70%、20%、10%)
ratio1 = 0.7
ratio2 = 0.2
s = np.int(num_example * ratio1)
s2 = np.int(num_example * ratio2)+s
X_train = data[:s]
y_train = label[:s]
fpaths_train = fpaths[:s] 
X_valid = data[s:s2]
y_valid = label[s:s2]
fpaths_valid = fpaths[s:s2] 

X_test = data[s2:]
y_test = label[s2:]
fpaths_tset = fpaths[s2:] 


X_train = X_train.astype('float32') / 255.
X_valid = X_valid.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
# print(len(x_train),len(y_train),len(x_val),len(y_val)) #800 800 200 200
# print(y_val)
print("Train", X_train.shape, y_train.shape)
print("Valid", X_valid.shape, y_valid.shape)
print("Test", X_test.shape, y_test.shape)


# In[6]:


# 建立CNN模型
# input_shape=np.array(tmp_img_rs).shape
input_shape=(150,150,3)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',activation='relu', input_shape=input_shape))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3),activation='relu', padding='same'))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3),activation='relu', padding='same'))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(256,activation='relu'))
# model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(128,activation='relu'))
# model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(num_classes, activation='softmax'))  #記得更改output數量
          
model.summary() # 秀出模型架構


# In[7]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False   # randomly flip images
            ) 

datagen.fit(X_train)


# In[8]:


# Model Train

batch_size = 24
epochs = 50
lr = 0.01

# 讓我們先配置一個常用的組合來作為後續優化的基準點
otm = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
#otm = Adam(lr=lr, decay=1e-6)
model.compile(loss='categorical_crossentropy',
             optimizer=otm,
             metrics=['accuracy'])

# earlystop
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# 保存在訓練過程中比較好的模型
filepath="model-dtaug.h5"
# 保留"val_acc"最好的那個模型
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')

#learning rate decay
def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

# callbacks_list = [LearningRateScheduler(lr_schedule) ,checkpoint ,earlystop ]

# history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
# #                             steps_per_epoch=X_train.shape[0] // (batch_size /3) ,
#                             epochs=epochs,
#                             validation_data=(X_valid, y_valid),
#                             #workers=2, cpu count
#                             callbacks=callbacks_list)


from keras.callbacks import ReduceLROnPlateau
learning_rate_function = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)
history = model.fit(X_train, y_train, batch_size=16, epochs=40, verbose=1, validation_data=(X_valid, y_valid),callbacks=[learning_rate_function])


# In[10]:


# 透過趨勢圖來觀察訓練與驗證的走向 

import matplotlib.pyplot as plt

def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics),'-o')
    plt.plot(history.history.get(val_metrics),'-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
    
plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_train_history(history, 'loss','val_loss')

plt.subplot(1,2,2)
plot_train_history(history, 'accuracy','val_accuracy')
plt.savefig('./his_fig/0105.jpg')
plt.show()


# In[11]:


#查看驗證集loss & accurancy
score = model.evaluate(X_test, y_test, verbose=1)
print(score)


# In[12]:


from keras.models import load_model

# 把訓練時val_loss最小的模型載入
# model = load_model('D:/dataset/model/cnn_model_real_yn_0731.h5')

# 預測
y_pred = model.predict_classes(X_test)


# In[14]:


#將模型存檔
model.save('./model/0105.h5')


# In[15]:


# len(y_pred)


# In[16]:


#將正確答案從one_hot encoding轉為一般label
y_ans = np.zeros(len(y_pred))
for i in range(len(y_pred)):
    y_ans[i]=np.argmax(y_test[i], axis=None, out=None)
print(y_ans)


# In[13]:


#印出預測資料、實際答案並計算準確度
count=0
for i in range(len(y_pred)):
    print(y_pred[i],y_ans[i])
    if(y_pred[i]==y_ans[i]):
        count+=1
# print()
print(count/len(y_pred))  #準確度


# In[19]:


#輸出混淆矩陣
# from pandas_ml import ConfusionMatrix
# cm = ConfusionMatrix(y_ans,y_pred)
# cm
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_ans,y_pred)
print(cm)
# In[18]:


#查看label數字與種類的對照表
print(os.listdir(path))


# %%
