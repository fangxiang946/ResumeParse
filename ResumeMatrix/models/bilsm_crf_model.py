from keras.layers import Input, Dense, LSTM, Embedding, recurrent, Bidirectional, Flatten, Dropout
from keras.models import Model
from keras.initializers import Constant
import pickle
import sys

sys.path.append('..')
from keras_contrib.layers import CRF

Epoch = 1
BATCH_SIZE = 100
BiRNN_UNITS = 100

def get_model(x_train, y_train, embedding_mat):
    inputSize, inputLength = x_train.shape
    #print('inputLength=%s' % inputLength)
    inputDim, outputDim = embedding_mat.shape
    #print('inputDim=%s,outputDim=%s' % (inputDim, outputDim))
    _, class_num = y_train.shape
    #print('class_num=%s' % class_num)

    with open('../ckpt/embedding_config.pkl', 'wb') as outp:
        pickle.dump((inputLength, inputDim, outputDim,class_num), outp)

    # define model
    inputs = Input(shape=(inputLength,))
    x = Embedding(inputDim, outputDim,embeddings_initializer=Constant(embedding_mat), trainable=False)(inputs)  # mask_zero=True
    x = Bidirectional(LSTM(128))(x)
    x = Dropout(0.3)(x)
    preds = Dense(class_num, activation='softmax')(x)
    model = Model(inputs,preds)
    model.compile('adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    return model

def Init_model():
    with open('../ckpt/embedding_config.pkl', 'rb') as inp:
        (inputLength, inputDim, outputDim,class_num) = pickle.load(inp)

    # define model
    inputs = Input(shape=(inputLength,))
    x = Embedding(inputDim, outputDim, trainable=False)(inputs)  # mask_zero=True
    x = Bidirectional(LSTM(128))(x)
    x = Dropout(0.3)(x)
    preds = Dense(class_num, activation='softmax')(x)
    model = Model(inputs, preds)
    model.compile('adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    return model


