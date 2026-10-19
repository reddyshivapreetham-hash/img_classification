import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path="C:\\Users\\User\\Desktop\\python codes\\ML\\testdata"

datagen=ImageDataGenerator(rescale=1./255,validation_split=0.2)

#training data
train_data=datagen.flow_from_directory(dataset_path,target_size=(128,128),
                                       batch_size=32,class_mode='categorical',
                                       subset='training')
val_data=datagen.flow_from_directory(dataset_path,target_size=(128,128),
                                     batch_size=32,class_mode='categorical',
                                     subset='validation')

#building a CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

model=Sequential()

model.add(Conv2D(32,(3,3),
                 activation='relu',
                 input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))


model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))


model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_data.num_classes,activation='softmax')) #otherwise dense(3 because we have three classes)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.fit(train_data,validation_data=val_data,epochs=20)

#classfying a new image !!
'''
import numpy as np
from tensorflow.keras.preprocessing import image
img_path='/content/orangeeeeeeeeeeeee.jpg'
img=image.load_img(img_path,target_size=(128,128))
img_array=image.img_to_array(img)/255.0
img_array=np.expand_dims(img_array,axis=0)

pred=model.predict(img_array)
#print(pred)
predicted_class=np.argmax(pred[0])

class_indices=train_data.class_indices
class_name=list(class_indices.keys())
print("predicted Class : ",class_name[predicted_class])'''

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
imgg_path="C:\\Users\\User\\Desktop\\python codes\\ML\\tdata\\Mandarin-Orange-581b2ccb3df78cc2e81b59a0.jpg"#'/content/Mandarin-Orange-581b2ccb3df78cc2e81b59a0.jpg'
img=image.load_img(imgg_path,target_size=(128,128))
img_array=image.img_to_array(img)/255.0
img_array=np.expand_dims(img_array,axis=0)
predictions=model.predict(img_array)
predicted_classes=np.argmax(predictions[0])

class_labels=list(train_data.class_indices.keys())
predicted_classes=class_labels[predicted_classes]
plt.imshow(image.load_img(imgg_path))
plt.title(f"predicted : {predicted_classes}")
plt.axis('off')
plt.show()

#model.save("fruite_classification.h5")

