import sys

import keras
import numpy as np
from keras.preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MaxAbsScaler



from DL_my_reddit_project.prepare_data import NUMBER_OF_LANGUAGES, Data_preparation, NUMBER_OF_FAMILIES, NUMBER_OF_BINARY, SPELLING

from keras import models
from keras.callbacks import EarlyStopping
from DL_my_reddit_project.LSTM_model_with_attention import create_lstm_model_with_attention
from keras.utils import plot_model
from keras.backend import manual_variable_initialization

from RedditConfig import RedditConfig
from numpy.random import seed

from tensorflow import set_random_seed

word_vec_size = 300
max_chunk_size = 1500 
attention_vec_dim = 300

def Build_and_train_LSTM_classifier():
    config = RedditConfig(sys.argv[1])

    dp = Data_preparation()
    X_train, y_train, X_test, y_test, number_of_classes, spell_X_train, spell_X_test = dp.prepare_data(config)

    vocabulary_size = dp.get_vocabulary_size(X_train, X_test)

    X_train = sequence.pad_sequences(X_train, maxlen=max_chunk_size, truncating='post', padding='post')
    X_test = sequence.pad_sequences(X_test, maxlen=max_chunk_size, truncating='post', padding='post')

    y_train = keras.utils.to_categorical(y_train, num_classes=number_of_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=number_of_classes)

    # split the training data into a training set and a validation set
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    data = X_train[indices]
    if SPELLING:
        spell_data = spell_X_train[indices]
    labels = y_train[indices]
    nb_validation_samples = int(0.1 * data.shape[0])

    X_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    X_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    if SPELLING:
        spell_X_train = spell_data[:-nb_validation_samples]
        spell_X_val = spell_data[-nb_validation_samples:]

    model = create_lstm_model_with_attention(max_chunk_size, vocabulary_size, word_vec_size, number_of_classes, SPELLING)
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    batch_size = 32
    seed(1)
    set_random_seed(2)
    my_callbacks = [EarlyStopping(monitor='val_acc', patience=3,mode=max)]
    if SPELLING:
        model.fit([X_train, spell_X_train], [y_train,y_train],validation_data=([X_val, spell_X_val], [y_val, y_val]), epochs=10, batch_size=batch_size, shuffle=True,
                  verbose=1, callbacks=my_callbacks)
        print("score on training data: ")
        scores = model.evaluate([X_train, spell_X_train], y_train, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        print("scores on test data:")
        scores = model.evaluate([X_test, spell_X_test], y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        evaluations = model.predict([X_test, spell_X_test])
    else:
        model.fit(X_train, y_train,validation_data=(X_val, y_val), epochs=14, batch_size=batch_size, shuffle=True, verbose=1, callbacks=my_callbacks)
        print("score on training data: ")
        scores = model.evaluate(X_train, y_train, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        print("scores on test data:")
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        evaluations = model.predict(X_test)
    cm = confusion_matrix(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), evaluations)))
    y_test_index = np.argmax(y_test, axis=1) # Convert one-hot to index
    y_predicted_index = np.argmax(evaluations, axis=1)
    #print(cm)
    print(classification_report(y_test_index, y_predicted_index))
	
if __name__ == '__main__':
    Build_and_train_LSTM_classifier()

