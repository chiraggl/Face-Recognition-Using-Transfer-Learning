#!/usr/bin/env python
# coding: utf-8

# ### Lets Create Our Own Dataset of faces

# In[1]:


#To generate Training Dataset

import cv2

# Initialize Webcam
cap = cv2.VideoCapture(0)

#Load Haarcascade Frontal Face Classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Function returns cropped face
def face_extractor(photo):
    gray_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_photo)
    
    if faces is ():
        return None
    
    else:
        # Crop all faces found
        for (x,y,w,h) in faces:
            cropped_face = photo[y:y+h, x:x+w]
        
        return cropped_face


count = 0

# Collect 100 samples of your face from webcam input
while True:
    status,photo = cap.read()
    
    if face_extractor(photo) is not None:
        count += 1
        face = cv2.resize(face_extractor(photo), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = '/home/chiraggl/tlfr/faces/train/chirag/face' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")


# In[2]:


#To Generate Test Dataset

import cv2

# Initialize Webcam
cap = cv2.VideoCapture(0)

#Load Haarcascade Frontal Face Classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Function returns cropped face
def face_extractor(photo):
    gray_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_photo)
    
    if faces is ():
        return None
    
    else:
        # Crop all faces found
        for (x,y,w,h) in faces:
            cropped_face = photo[y:y+h, x:x+w]
        
        return cropped_face


count = 0

# Collect 100 samples of your face from webcam input
while True:
    status,photo = cap.read()
    
    if face_extractor(photo) is not None:
        count += 1
        face = cv2.resize(face_extractor(photo), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = '/home/chiraggl/tlfr/faces/test/chirag/face' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        pass

    if cv2.waitKey(1) == 13 or count == 70: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")


# ### Load the VGG16 Model

# In[3]:


from keras.applications import vgg16

# VGG16 was designed to work on 224 x 224 pixel input images sizes
img_rows, img_cols = 224, 224 

# Loads the VGG16 model without the top or FC layers
model = vgg16.VGG16(weights='imagenet', include_top = False, input_shape=(img_rows,img_cols,3))

# Let's print our layers 
for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# ### Freeze the Layers

# In[4]:


# Here we freeze the layers 
# Layers are set to trainable as True by default
for layer in model.layers:
    layer.trainable = False
    
# Let's print our layers 
for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# ### Lets create a function to add new Layers on top

# In[5]:


def addTopModel(bottom_model, num_classes):
    #creates the top or head of the model that will be placed ontop of the bottom layers

    top_model = bottom_model.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model


# ### Adding our FC Head back onto VGG16 model

# In[6]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# Set the number of classes
num_classes = 1

FC_Head = addTopModel(model, num_classes)

modelnew = Model(inputs = model.input, outputs = FC_Head)

print(modelnew.summary())


# ### Apply Image Augmentation

# In[7]:


from keras_preprocessing.image import ImageDataGenerator

train_data_dir = '/home/chiraggl/tlfr/faces/train/'
validation_data_dir = '/home/chiraggl/tlfr/faces/test/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Set the Batch Size according to your system.
batch_size = 16

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


# ### Training Our Model

# In[11]:


from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("facedetect.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# We use a very small learning rate 
modelnew.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 100
nb_validation_samples = 70

# We only train 5 EPOCHS 
epochs = 5
batch_size = 16

history = modelnew.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

modelnew.save('facedetect.h5')





