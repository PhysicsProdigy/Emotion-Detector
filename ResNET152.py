from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform

from livelossplot import PlotLossesKerasTF

import numpy as np
import tensorflow as tf

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=7000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

def identity_block(img, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    img_copy = img

    img = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', name=conv_name_base + '2a',
                 kernel_initializer = glorot_uniform(seed = 0))(img)
    img = BatchNormalization(axis = 3, name = bn_name_base + '2a')(img)
    img = Activation('relu')(img)

    img = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(img)
    img = BatchNormalization(axis=3, name=bn_name_base + '2b')(img)
    img = Activation('relu')(img)

    img = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(img)
    img = BatchNormalization(axis=3, name=bn_name_base + '2c')(img)

    img = Add()([img, img_copy])
    img = Activation('relu')(img)

    return img

def convolutional_block(img, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    img_copy = img

    img = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(img)
    img = BatchNormalization(axis=3, name=bn_name_base + '2a')(img)
    img = Activation('relu')(img)

    img = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(img)
    img = BatchNormalization(axis=3, name=bn_name_base + '2b')(img)
    img = Activation('relu')(img)

    img = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(img)
    img = BatchNormalization(axis=3, name=bn_name_base + '2c')(img)

    img_copy = Conv2D(F3, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(img_copy)
    img_copy = BatchNormalization(axis=3, name=bn_name_base + '1')(img_copy)

    img = Add()([img, img_copy])
    img = Activation('relu')(img)

    return img


def ResNet152(input_shape=(64, 64, 1), classes=7):
    # Define the input as a tensor with shape input_shape
    img_Input = Input(input_shape)

    # Zero-Padding
    img = ZeroPadding2D((3, 3))(img_Input)

    # Stage 1
    img = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(img)
    img = BatchNormalization(axis=3, name='bn_conv1')(img)
    img = Activation('relu')(img)
    img = MaxPooling2D((3, 3), strides=(2, 2))(img)

    # Stage 2
    img = convolutional_block(img, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    img = identity_block(img, 3, [64, 64, 256], stage=2, block='b')
    img = identity_block(img, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    img = convolutional_block(img, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    img = identity_block(img, 3, [128, 128, 512], stage=3, block='b')
    img = identity_block(img, 3, [128, 128, 512], stage=3, block='c')
    img = identity_block(img, 3, [128, 128, 512], stage=3, block='d')
    img = identity_block(img, 3, [128, 128, 512], stage=3, block='e')
    img = identity_block(img, 3, [128, 128, 512], stage=3, block='f')
    img = identity_block(img, 3, [128, 128, 512], stage=3, block='g')
    img = identity_block(img, 3, [128, 128, 512], stage=3, block='h')


    # Stage 4
    img = convolutional_block(img, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='b')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='c')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='d')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='e')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='f')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='g')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='h')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='i')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='j')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='k')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='l')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='m')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='n')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='o')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='p')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='q')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='r')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='s')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='t')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='u')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='v')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='w')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='x')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='y')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='z')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='aa')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='ab')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='ac')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='ad')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='ae')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='af')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='ag')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='ah')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='ai')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='aj')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='ak')
    img = identity_block(img, 3, [256, 256, 1024], stage=4, block='al')


    # Stage 5
    img = convolutional_block(img, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    img = identity_block(img, 3, [512, 512, 2048], stage=5, block='b')
    img = identity_block(img, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL . Use "img = AveragePooling2D(...)(img)"
    img = AveragePooling2D()(img)

    # output layer
    img = Flatten()(img)

    # Fully connected layer 1st layer
    img = Dense(256)(img)
    img = BatchNormalization()(img)
    img = Activation('relu')(img)

    # Fully connected layer 2nd layer
    img = Dense(512)(img)
    img = BatchNormalization()(img)
    img = Activation('relu')(img)

    img = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(img)

    # Create model
    model = Model(inputs=img_Input, outputs=img, name='ResNet50')

    return model

datagen_train = ImageDataGenerator(horizontal_flip=True)

train_generator = datagen_train.flow_from_directory("train/",
                                                    target_size=(64,64),
                                                    color_mode="grayscale",
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    shuffle=True)
#Generating validation batches
datagen_validation = ImageDataGenerator(horizontal_flip=True)

validation_generator = datagen_validation.flow_from_directory("test/",
                                                    target_size=(64,64),
                                                    color_mode="grayscale",
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    shuffle=False)

model = ResNet152(input_shape=(64,64,1), classes = 7)
opt = Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 15
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size
datagen_train = ImageDataGenerator(horizontal_flip=True)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, mode='auto')
checkpoint = ModelCheckpoint("resNet152_Weights.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
callbacks = [PlotLossesKerasTF(),checkpoint, reduce_lr]

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    callbacks=callbacks
)
#Save model into a JSON file.
model_json = model.to_json()
with open("resNet152.json", "w") as json_file:
    json_file.write(model_json)
    print("Sucess!")


