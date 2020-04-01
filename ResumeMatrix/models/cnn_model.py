from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Embedding
from keras.optimizers import RMSprop

def get_model(x_train, y_train, embedding_mat):
    inputSize, inputLength = x_train.shape
    print('inputLength=%s' % inputLength)
    inputDim, outputDim = embedding_mat.shape
    print('inputDim=%s,outputDim=%s' % (inputDim, outputDim))
    _, class_num = y_train.shape
    print('class_num=%s' % class_num)

    inputs = Input(shape=(inputLength,))
    x = Embedding(inputDim, outputDim, weights=[embedding_mat], trainable=False, mask_zero=True)(inputs)

    x = Conv2D(32, (5, 5), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Dropout(0.3)(x)
    x = Flatten()(x)
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