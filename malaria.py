
"""
Created on Fri May  1 11:24:32 2020

@author: abt
"""
"""Gerekli kütüphaneler"""
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

#Resimlerin boyutu
goruntu_genislik, goruntu_yukseklik = 299, 299

#eğitim verisinin lokasyonu
egitim_seti_dizin = 'veriseti/egitim' 
#doğrulama(validation) verisinin lokasyonu
dogrulama_seti_dizin = 'veriseti/dogrulama' 

# samples_per_epoch'u belirlemek için kullanılan örnek sayısı
nb_train_samples = 65
nb_validation_samples = 10

# eğitim verisinden geçiş sayısı (tur)
epochs = 20
#aynı anda işlenen görüntü sayısı
batch_size = 5  

""" Veri ön işleme Data Augmentation  """
egitim_veri_uretici = ImageDataGenerator(
        rescale=1./255,            # piksel değerlerini [0,1] normalize eder
        shear_range=0.2,      
        zoom_range=0.2,    
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)  


dogrulama_veri_uretici = ImageDataGenerator(
         rescale=1./255)      # piksel değerlerini [0,1] normalize eder


train_generator = egitim_veri_uretici.flow_from_directory(
    egitim_seti_dizin,
    target_size=(goruntu_yukseklik, goruntu_genislik),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = dogrulama_veri_uretici.flow_from_directory(
    dogrulama_seti_dizin,
    target_size=(goruntu_yukseklik, goruntu_genislik),
    batch_size=batch_size,
    class_mode='binary')

from keras.applications.inception_v3 import InceptionV3

deneme_model= InceptionV3()

print(deneme_model.summary())

 
temel_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(goruntu_genislik, goruntu_yukseklik, 3))

ust_model = Sequential()
ust_model.add(GlobalAveragePooling2D(input_shape=temel_model.output_shape[1:], data_format=None)),  
ust_model.add(Dense(256, activation='relu'))
ust_model.add(Dropout(0.5))
ust_model.add(Dense(1, activation='sigmoid')) 

model = Model(inputs=temel_model.input, outputs=ust_model(temel_model.output))

model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])


gecmis = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

"""Grafik çizme  """

import matplotlib.pyplot as plt

print(gecmis.history.keys())

plt.figure()
plt.plot(gecmis.history['accuracy'], 'orange', label='Egitim Dogruluk')
plt.plot(gecmis.history['val_accuracy'], 'blue', label='Dogrulama Dogruluk')
plt.plot(gecmis.history['loss'], 'red', label='Egitim Kayıp')
plt.plot(gecmis.history['val_loss'], 'green', label='Dogrulama Kayıp')
plt.legend()
plt.show()


""" Test verisini kullanarak modele tahmin yaptırma"""

import numpy as np
from keras.preprocessing import image

goruntu_dizin ='veriseti/test/pozitif/1.png' 
goruntu_dizin2='veriseti/test/pozitif/2.png'  
goruntu_dizin3='veriseti/test/pozitif/3.png' 
goruntu_dizin4='veriseti/test/pozitif/4.png' 
goruntu_dizin5='veriseti/test/pozitif/5.png' 
goruntu_dizin6='veriseti/test/negatif/1.png' 
goruntu_dizin7='veriseti/test/negatif/2.png' 
goruntu_dizin8='veriseti/test/negatif/3.png' 
goruntu_dizin9='veriseti/test/negatif/4.png' 
goruntu_dizin10='veriseti/test/negatif/5.png' 

goruntu = image.load_img(goruntu_dizin, target_size=(goruntu_genislik, goruntu_yukseklik))
goruntu2 = image.load_img(goruntu_dizin2, target_size=(goruntu_genislik, goruntu_yukseklik))
goruntu3 = image.load_img(goruntu_dizin3, target_size=(goruntu_genislik, goruntu_yukseklik))
goruntu4 = image.load_img(goruntu_dizin4, target_size=(goruntu_genislik, goruntu_yukseklik))
goruntu5 = image.load_img(goruntu_dizin5, target_size=(goruntu_genislik, goruntu_yukseklik))
goruntu6 = image.load_img(goruntu_dizin6, target_size=(goruntu_genislik, goruntu_yukseklik))
goruntu7 = image.load_img(goruntu_dizin7, target_size=(goruntu_genislik, goruntu_yukseklik))
goruntu8 = image.load_img(goruntu_dizin8, target_size=(goruntu_genislik, goruntu_yukseklik))
goruntu9 = image.load_img(goruntu_dizin9, target_size=(goruntu_genislik, goruntu_yukseklik))
goruntu10 = image.load_img(goruntu_dizin10, target_size=(goruntu_genislik, goruntu_yukseklik))

plt.imshow(goruntu)
plt.show()
goruntu= image.img_to_array(goruntu)
x = np.expand_dims(goruntu, axis=0) * 1./255
score = model.predict(x)
print('Tahmin:', score, 'Negatif' if score < 0.5 else 'Pozitif')


plt.imshow(goruntu2)
plt.show()
goruntu = image.img_to_array(goruntu2)
x = np.expand_dims(goruntu2, axis=0) * 1./255
score2 = model.predict(x)
print('Tahmin:', score2, 'Negatif' if score2 < 0.5 else 'Pozitif ')


plt.imshow(goruntu3)
plt.show()
goruntu= image.img_to_array(goruntu3)
x = np.expand_dims(goruntu3, axis=0) * 1./255
score = model.predict(x)
print('Tahmin:', score, 'Negatif' if score < 0.5 else 'Pozitif')


plt.imshow(goruntu4)
plt.show()
goruntu= image.img_to_array(goruntu4)
x = np.expand_dims(goruntu4, axis=0) * 1./255
score = model.predict(x)
print('Tahmin:', score, 'Negatif' if score < 0.5 else 'Pozitif')


plt.imshow(goruntu5)
plt.show()
goruntu= image.img_to_array(goruntu5)
x = np.expand_dims(goruntu5, axis=0) * 1./255
score = model.predict(x)
print('Tahmin:', score, 'Negatif' if score < 0.5 else 'Pozitif')


plt.imshow(goruntu6)
plt.show()
goruntu= image.img_to_array(goruntu6)
x = np.expand_dims(goruntu6, axis=0) * 1./255
score = model.predict(x)
print('Tahmin:', score, 'Negatif' if score < 0.5 else 'Pozitif')


plt.imshow(goruntu7)
plt.show()
goruntu= image.img_to_array(goruntu7)
x = np.expand_dims(goruntu7, axis=0) * 1./255
score = model.predict(x)
print('Tahmin:', score, 'Negatif' if score < 0.5 else 'Pozitif')


plt.imshow(goruntu8)
plt.show()
goruntu= image.img_to_array(goruntu8)
x = np.expand_dims(goruntu8, axis=0) * 1./255
score = model.predict(x)
print('Tahmin:', score, 'Negatif' if score < 0.5 else 'Pozitif')


plt.imshow(goruntu9)
plt.show()
goruntu= image.img_to_array(goruntu9)
x = np.expand_dims(goruntu9, axis=0) * 1./255
score = model.predict(x)
print('Tahmin:', score, 'Negatif' if score < 0.5 else 'Pozitif')


plt.imshow(goruntu10)
plt.show()
goruntu= image.img_to_array(goruntu10)
x = np.expand_dims(goruntu10, axis=0) * 1./255
score = model.predict(x)
print('Tahmin:', score, 'Negatif' if score < 0.5 else 'Pozitif')

