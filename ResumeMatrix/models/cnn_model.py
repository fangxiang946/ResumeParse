from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D,Embedding,GlobalMaxPool1D
from keras.optimizers import RMSprop
from keras.initializers import Constant

def get_model(x_train, y_train, embedding_mat):
    inputSize, inputLength = x_train.shape
    #print('inputLength=%s' % inputLength)
    inputDim, outputDim = embedding_mat.shape
    #print('inputDim=%s,outputDim=%s' % (inputDim, outputDim))
    _, class_num = y_train.shape
    #print('class_num=%s' % class_num)

    inputs = Input(shape=(inputLength,))
    x = Embedding(inputDim, outputDim, embeddings_initializer=Constant(embedding_mat), trainable=False)(inputs)  # mask_zero=True

    x = Conv1D(64,5, activation='relu')(x)
    x = MaxPooling1D(5)(x)

    x = Conv1D(64, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)

    x = Conv1D(64, 5, activation='relu')(x)
    x = GlobalMaxPool1D()(x)

    x = Dropout(0.3)(x)
    #x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(class_num, activation='softmax')(x)

    model = Model(inputs, outputs)
    # initiate RMSprop optimizer
    opt = RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.summary()
    return model