import os

import matplotlib
from pasta.augment import inline
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip

from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

#Загрузка данных
#------------------------------------------------------------------------->
#Чтение данных из формата gzip в массив типа NumPy float32
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28,28)
        return data

train_data = extract_data('train-images-idx3-ubyte.gz', 60000)
test_data = extract_data('t10k-images-idx3-ubyte.gz', 10000)

#Чтение символов (labels) из имеющегося массиваа данных
def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

train_labels = extract_labels('train-labels-idx1-ubyte.gz',60000)
test_labels = extract_labels('t10k-labels-idx1-ubyte.gz',10000)
#------------------------------------------------------------------------->

#Словарь
label_dict = {
 0: 'A',
 1: 'B',
 2: 'C',
 3: 'D',
 4: 'E',
 5: 'F',
 6: 'G',
 7: 'H',
 8: 'I',
 9: 'J',
}

# Посмотреть размер изображения (размер обучающей выборки, и размерность матрицы)
# Размер тренировочного изображения
print("Training set (images) shape: {shape}".format(shape=train_data.shape))
#Размер тестового изображения
print("Test set (images) shape: {shape}".format(shape=test_data.shape))
#----------------------------------------------------------------------------------->
# Пробное отображение изображений из выборок
# plt.figure(figsize=[5,5])
# # Отображение первого изоюражения из тренировочной выборки
# plt.subplot(121)
# curr_img = np.reshape(train_data[0], (28,28))
# curr_lbl = train_labels[0]
# plt.imshow(curr_img, cmap='gray')
# plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
# # Отображение первого изоюражения из тестовой выборки
# plt.subplot(122)
# curr_img = np.reshape(test_data[0], (28,28))
# curr_lbl = test_labels[0]
# plt.imshow(curr_img, cmap='gray')
# plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
#----------------------------------------------------------------------------------->
#Предварительная обработка данных
#Конвертация имеющихся данных из матрицы 28x28 в матрицу 28x28x1
train_data = train_data.reshape(-1, 28,28, 1)
test_data = test_data.reshape(-1, 28,28, 1)
train_data.shape, test_data.shape

#Нужно проверить, что тип имеющихся данных NumPy float32,
print(train_data.dtype, test_data.dtype)
#но мы уже преобразовали данные

#Изменить масштаб данных
np.max(train_data), np.max(test_data)

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)
np.max(train_data), np.max(test_data)
print(np.max(train_data), np.max(test_data))
#----------------------------------------------------------------------------------->
#Разделение данных
# обучающие данные 80%, проверочные 20%
train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                             train_data,
                                                             test_size=0.2,
                                                             random_state=13)
#----------------------------------------------------------------------------------->
#АВТОЭНКОДЕР
batch_size = 128
epochs = 50 #за столько произойдет обучение
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))

def autoencoder(input_img):
    #Кодировщик
    #1 слой - 32 фильтра 3x3,+ слой понижения дискретизации
    #2 слой - 64 фильтра 3x3, + слой понижения дискретизации
    #3 слой 128 фильтров 3x3

    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #Декодер
    #1 слой - 128 фильтров 3x3,+ слой повышения дискретизации
    #2 слой - 64 фильтра 3x3, + слой повышения дискретизации
    #3 слой 1 фильтров 3x3
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

#loss - тип потери (среднеквадратичная ошибка
autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='binary_crossentropy', optimizer = RMSprop())

# Если надо посмотреть значения слоев в автоэнкодере
# autoencoder.summary()
#---------------------------------------------------------------------------------->
#Обучение модели
autoencoder_train = autoencoder.fit(train_X,
                                    train_ground,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(valid_X, valid_ground))
#---------------------------------------------------------------------------------->
#График потерь
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#---------------------------------------------------------------------------------->
#Прогнозирование
pred = autoencoder.predict(test_data)
print(pred.shape)

plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(test_data[i, ..., 0], cmap='gray')
    curr_lbl = test_labels[i]
    plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

plt.show()

plt.figure(figsize=(20, 4))

print("Reconstruction of Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')

plt.show()
