import sys

import keras
import tensorflow as tf
import tensorflow_hub as hub
from keras import Sequential, optimizers, Model
from keras import layers
from keras.engine import Layer
from keras.layers import Embedding, LSTM, Dense, Activation, TimeDistributed, Flatten, RepeatVector, Permute, merge, \
    Lambda, K, concatenate, Multiply, multiply, Bidirectional
from keras.utils import plot_model
from keras.preprocessing import sequence
from DL_my_reddit_project.prepare_data import NUMBER_OF_LANGUAGES, Data_preparation, NUMBER_OF_FAMILIES, NUMBER_OF_BINARY
from keras.layers import Dropout
from keras import models
from keras import Input
from keras.optimizers import SGD, Adam
from keras.models import load_model

class NonMasking(Layer):
    def __init__(self, ** kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)


    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        #  do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


def create_lstm_model_with_attention(max_chunk_size, vocab_size, embedding_size, num_of_classes, spelling=False):
    units = embedding_size
    max_chunk_length = max_chunk_size
    vocab_size = vocab_size#embeddings.shape[0]
    embedding_size = embedding_size#embeddings.shape[1]#word_vec_size=32
    trainable = True
    masking = False

    _input = Input(shape=[max_chunk_length], dtype='int32', name='input')
    _spelling_input = Input(shape=[458], name='spelling_features_input')
    # get the embedding layer
    embedded = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_chunk_length,
        mask_zero=masking, trainable=trainable
    )(_input)
    if masking:
        no_masking = NonMasking()(embedded)
        activations = LSTM(units, return_sequences=True, dropout=0.6)(no_masking)
    else:
        activations = LSTM(units, return_sequences=True, dropout=0.6)(embedded)


    # compute importance for each step
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)
    attention = Permute([2, 1])(attention)

    # apply the attention
    chunk_representation = multiply([activations, attention])  
    chunk_representation = Lambda(lambda xin: K.sum(xin, axis=1))(chunk_representation)
    if spelling:
        aux_output = Dense(num_of_classes, activation='softmax', name='aux_output')(chunk_representation)

        spelling_combined_layer = concatenate([chunk_representation, _spelling_input])
        ffnn = Dense(64, activation='relu')(spelling_combined_layer)
        probabilities = Dense(num_of_classes, activation='softmax')(ffnn)
        model = Model(inputs=[_input,_spelling_input], outputs=[probabilities,aux_output])
    else:
        probabilities = Dense(num_of_classes, activation='softmax')(chunk_representation)
        model = Model(input=_input, output=probabilities)

    opt = Adam(lr=0.003)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
