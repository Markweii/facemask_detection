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
from keras.layers import Dense, Conv2D, Input, LeakyReLU

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
                                    target_size=(128, 128),
                                    class_mode='sparse', 
                                    )

val_generator = train_datagen.flow_from_directory(
                                    directory=training_path,
                                    subset='validation',
                                    batch_size=32,                                    
                                    target_size=(128, 128),
                                    class_mode='sparse', 
                                    )

test_generator = test_datagen.flow_from_directory(
                                    directory=testing_path,
                                    batch_size=75,                                    
                                    target_size=(128, 128),
                                    class_mode='sparse', 
                                    )

#------------------------------------------------------------------------------------------------------- 
#Transfer Learning
#------------------------------------------------------------------------------------------------------- 

base_model = tf.keras.applications.densenet.DenseNet121(
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True


#------------------------------------------------------------------------------------------------------- 
#模型之函數式設置
#------------------------------------------------------------------------------------------------------- 

inputs = Input(shape=(128,128, 3), name="img")
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(1000, activation=LeakyReLU(alpha=0.05))(x)
x = Dense(500, activation=LeakyReLU(alpha=0.0))(x)
outputs = Dense(3, activation="softmax")(x)

trans_densenet121 = Model(inputs, outputs, name="trans_densenet121")
trans_densenet121.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            optimizer='adam', metrics=['accuracy'])


#------------------------------------------------------------------------------------------------------- 
#訓練及顯示
#------------------------------------------------------------------------------------------------------- 

# 顯示類神經網路架構
trans_densenet121.summary()
#畫出類神經網路架構 存到trans_densenet121.png裡
tf.keras.utils.plot_model(trans_densenet121,to_file="trans_densenet121.png",dpi=96) 

#顯示訓練過程
history = trans_densenet121.fit_generator(
                              train_generator , 
                              epochs=50,
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
trans_densenet121.save("dense_model.h5")
#使用已儲存之模型
# trans_densenet121 = load_model("dense_model.h5")


#------------------------------------------------------------------------------------------------------- 
#Evaluation
#------------------------------------------------------------------------------------------------------- 

#EVL1: accuracy score
(X,y_true) = test_generator.next()
y_predictions = trans_densenet121.predict(X) #回傳各圖之三類機率，需以下式過濾出機率最大的類別
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









