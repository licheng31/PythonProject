import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import densenet
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
import os
import shutil
import json
from absl import flags
from absl import app


flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_integer('steps_per_epoch', 100, '')
flags.DEFINE_integer('epochs', 100, '')
flags.DEFINE_integer('validation_steps', 100, '')

flags.DEFINE_string('data_dir', default='./data/raw-data/', help='')
flags.DEFINE_string('logs', './logs', '')
#tf.app.flags.DEFINE_float('learning_rate', '0.001', '')
FLAGS = flags.FLAGS


def main(_):
    train_datagen = ImageDataGenerator(rescale=1./255)
    validate_datagen = ImageDataGenerator(rescale=1./255)

    train_dir = os.path.join(FLAGS.data_dir, 'train')
    validate_dir = os.path.join(FLAGS.data_dir, 'validation')

    #target_size = (224, 224)
    target_size = (299, 299)
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=target_size,
                                                        batch_size=FLAGS.batch_size,
                                                        color_mode='grayscale')
    #                                                    color_mode='rgb')
    validate_generator = validate_datagen.flow_from_directory(validate_dir,
                                                        target_size=target_size,
                                                        batch_size=FLAGS.batch_size,
                                                       color_mode='grayscale')
    #                                                    color_mode='rgb')


    #conv_base = inception_v3.InceptionV3(weights=None,
    #                                    include_top=True,
    #                                     input_shape=(299, 299, 3),
    #                                     classes=3755)

    model = inception_v3.InceptionV3(weights=None, include_top=True,
                        input_shape=(299, 299, 1), classes=240)

    #model = keras_alexnet.AlexNet(num_classses=240)
    #model = vgg19.VGG19(weights=None, include_top=True,
    #                    input_shape=(224, 224, 3), classes=240)
    #model = resnet.ResNet101(weights=None, include_top=True,
    #                 input_shape=(224, 224, 3), classes=240)
    #model = densenet.DenseNet121(weights=None, include_top=True,
    #                             input_shape=(224, 224, 3), classes=240)
    #model = densenet.DenseNet169(weights=None, include_top=True,
    #                             input_shape=(224, 224, 3), classes=240)
    #model = models.Sequential()
    #model.add(conv_base)
    #model_top = models.Sequential()
    #model.add(layers.GlobalAveragePooling2D(name='global_pool'))
    #model.add(layers.Dense(240, activation='softmax', name='softmax'))
    #model.add(model_top)
    #model = models.load_model('./keras_model/inceptionV3_byword_9626.h5')


    #learning_rate = FLAGS.learning_rate
    #opt = optimizers.RMSprop(lr=learning_rate, decay=0.5)
    #top5 = metrics.TopKCategoricalAccuracy(k=5, name='top_5_acc')
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    log_path = FLAGS.logs
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    #else:
    #    os.remove(log_path)
    #    os.mkdir(log_path)
    #write label file
    #train_lable = train_generator.class_indices
    #f = open('./keras_model/vgg19_240_label.txt', mode='w+')
    #json.dump(train_lable, f)
    #f.close()

    callback = [callbacks.TensorBoard(log_dir=log_path)]
    #callback[0].set_model(model)
    #callback.append(early_stopping)
    checkpoint = callbacks.ModelCheckpoint(filepath='./keras_model/inceptionV3_240_2021.h5',
                                           verbose=1, save_best_only=True, monitor='val_loss',
                                           save_weights_only=False)
    callback.append(checkpoint)
    model.fit_generator(train_generator,
                                  steps_per_epoch=FLAGS.steps_per_epoch,
                                  epochs=FLAGS.epochs,
                                  validation_data=validate_generator,
                                  validation_steps=FLAGS.validation_steps,
                                  workers=1,
                                  callbacks=callback)

    #model.save('./model/inceptionV3.h5')
    return 0

if __name__ == "__main__":
    app.run(main)
