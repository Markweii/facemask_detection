import os
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, Model
from keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, Add, SeparableConv2D
from keras.layers import BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import LeakyReLU, concatenate
from sklearn.model_selection import train_test_split

#路徑設定
training_path = 'C://Users//00332//Desktop//lab//moohai//Dataset'   #9351張圖
testing_path = 'C://Users//00332//Desktop//lab//moohai//Testset'    #75張圖


#------------------------------------------------------------------------------------------------------- 
#資料擴增
#-------------------------------------------------------------------------------------------------------
 
train_datagen = ImageDataGenerator( 
                                rescale= 1/255,
                                rotation_range=45,
                                horizontal_flip=True,                                 
                                shear_range=0.2, 
                                zoom_range=0.5,
                                fill_mode="nearest",
                                validation_split=0.2
                               )

test_datagen = ImageDataGenerator(rescale= 1/255)

#各輸出為一个生成 (x, y) 元组的 DirectoryIterator
#其中 x 是一个包含一批尺寸为 (batch_size, *target_size, channels)的圖像的 Numpy 數組，y 是對應標籤的Numpy數組
train_generator = train_datagen.flow_from_directory(
                                    directory=training_path,
                                    subset='training',                                   
                                    batch_size=32,                                    
                                    target_size=(32, 32),
                                    class_mode='sparse', 
                                    )

val_generator = train_datagen.flow_from_directory(
                                    directory=training_path,
                                    subset='validation',
                                    batch_size=32,                                    
                                    target_size=(32, 32),
                                    class_mode='sparse', 
                                    )

test_generator = test_datagen.flow_from_directory(
                                    directory=testing_path,
                                    batch_size=75,                                    
                                    target_size=(32, 32),
                                    class_mode='sparse', 
                                    )


#------------------------------------------------------------------------------------------------------- 
#定義DenseBlock
#------------------------------------------------------------------------------------------------------- 

#DenseNet-B結構。降低特徵數量，從而提升計算效率
def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
 
    # Bottleneck layers 採用1*1CONV減少計算量
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(bn_size*nb_filter, (1, 1), strides=(1,1), padding='same')(x)
 
    # Composite function
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (3,3), strides=(1,1), padding='same')(x)
 
    if drop_rate: x = Dropout(drop_rate)(x)
 
    return x


def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
 
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=3)
    return x


#DenseNet-C結構
#連結兩個相鄰的DenseBlock，壓縮特徵圖的數量。降低係數為compression
def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):
 
    nb_filter = int(x.shape.as_list()[-1]*compression)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (1, 1), strides=(1,1), padding='same')(x)
    if is_max != 0: x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    else: x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
 
    return x


#------------------------------------------------------------------------------------------------------- 
#模型之函數式設置
#------------------------------------------------------------------------------------------------------- 

growth_rate = 12

inputs = Input(shape=(32,32,3), name="img")
x = Conv2D(growth_rate*2, (3, 3), strides=1, padding='same')(inputs)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = TransitionLayer(x)
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = TransitionLayer(x)
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = TransitionLayer(x)
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1000, activation=LeakyReLU(alpha=0.05))(x)
x = Dense(500, activation=LeakyReLU(alpha=0.0))(x)
outputs = Dense(3, activation='softmax')(x)

self_defined = Model(inputs, outputs, name="self_defined")
self_defined.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            optimizer='adam', metrics=['accuracy'])


#------------------------------------------------------------------------------------------------------- 
#訓練及顯示
#------------------------------------------------------------------------------------------------------- 

# 顯示類神經網路架構
self_defined.summary()
#畫出類神經網路架構 存到self_defined.png裡
tf.keras.utils.plot_model(self_defined,to_file="self_defined.png",dpi=96) 

#訓練及顯示訓練過程
history = self_defined.fit_generator(
                              train_generator , 
                              epochs=35,
                              validation_data=val_generator,
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10),
                                         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=5)])


#顯示訓練損失歷史曲線
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()


#存訓練模型
self_defined.save("self_defined_model.h5")

#使用已儲存之模型 #改良版為improved_self_defined_model.h5，未改良版為self_defined_model
#self_defined = load_model("self_defined_model.h5") 


#------------------------------------------------------------------------------------------------------- 
#Evaluation
#------------------------------------------------------------------------------------------------------- 

#EVL1: accuracy score
(X,y_true) = test_generator.next()
y_predictions = self_defined.predict(X) #回傳各圖之三類機率，需以下式過濾出機率最大的類別
y_predictions = np.argmax(y_predictions, axis = 1)  
print(accuracy_score(y_true,y_predictions))


#EVL2: classification_report
print(classification_report(y_true, y_predictions))


#EVL3: confusion matrix
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']
titles_options = [("Confusion matrix", None),]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_predictions(
                                                    y_true,
                                                    y_predictions
                                                    )


plt.show()







