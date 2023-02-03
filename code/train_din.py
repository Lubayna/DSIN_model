import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (LSTM, Input, Dense, Embedding)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.compat.v1.ConfigProto()
tfconfig.gpu_options.allow_growth = True

input = Input(shape=(100,), dtype='int32', name='input')
x = Embedding(
    output_dim=512, input_dim=10000, input_length=100)(input)
x = LSTM(32)(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid', name='output')(x)
model = Model(inputs=[input], outputs=[output])
dot_img_file = '/tmp/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)