import argparse
import warnings

import mlflow
import mlflow.tensorflow
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from mlflow_callback import MlFlowCallback
 
IMAGE_SHAPE = (218, 178, 3)
TRAINING_SAMPLES = 8000
TEST_SAMPLES = 2000
#TRAINING_SAMPLES = 162770

ds = tfds.load('celeb_a', split=['train','test'], data_dir="data/")

def data_generator(
     batch_size,
     training_samples,
     validation_samples,
     train=True):

     def generator():
          index, samples = 0, training_samples if train else 1, validation_samples
          for sample in ds[index].take(samples).batch(batch_size).repeat():
               yield sample['image']/255, tf.map_fn(lambda label: 1 if label else 0, sample['attributes']['Smiling'], dtype=tf.int32)
     
     return generator
     
if __name__ == "__main__":
     warnings.filterwarnings('ignore')
     
     parser = argparse.ArgumentParser()
     parser.add_argument('--batch-size')
     parser.add_argument('--epochs')
     parser.add_argument('--convolutions')
     args = parser.parse_args()

     batch_size = int(args.batch_size)
     epochs = int(args.epochs)
     convolutions = int(args.convolutions)

     train_dataset = tf.data.Dataset.from_generator(
          data_generator(batch_size, train=True),
          (tf.uint8, tf.uint8),
          (tf.TensorShape([None,218,178,3]), tf.TensorShape([None])))

     test_dataset = tf.data.Dataset.from_generator(
          data_generator(batch_size, train=False),
          (tf.uint8, tf.uint8),
          (tf.TensorShape([None,218,178,3]), tf.TensorShape([None])))

     with mlflow.start_run():

          mlflow.tensorflow.autolog()

          model = Sequential()
          model.add(Input(shape=IMAGE_SHAPE))
          model.add(RandomFlip())

          for x in range(convolutions):
               model.add(Conv2D(32, (3,3), strides=(1,1), activation='relu'))
               model.add(MaxPooling2D())
               model.add(Dropout(0.5))

          model.add(Flatten())
          model.add(Dense(32, activation='relu'))
          model.add(Dense(1, activation='sigmoid'))

          model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
          model.fit(train_dataset,
               validation_data=test_dataset,
               epochs=epochs, 
               steps_per_epoch=TRAINING_SAMPLES/batch_size,
               validation_steps=TEST_SAMPLES/batch_size,
               callbacks=[MlFlowCallback()])