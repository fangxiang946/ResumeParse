import process_data
import models.bilsm_crf_model as lsmcrf
import models.cnn_model as cnn


EPOCHS = 3
BATCH_SIZE=200

def trainByLstm():
    x_train,y_train,x_val,y_val,embedding_mat = process_data.get_data(cnum=10000,test_size=0.1)

    model =lsmcrf.get_model(x_train,y_train,embedding_mat)

    model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS, validation_data=[x_val,y_val])
    model.save('../ckpt/crf.h5')


def trainByCnn():
    x_train, y_train, x_val, y_val, embedding_mat = process_data.get_data(cnum=10000, test_size=0.1)

    model = cnn.get_model(x_train, y_train, embedding_mat)

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=[x_val, y_val])
    model.save('../ckpt/cnn.h5')

if __name__ == '__main__':
    trainByCnn()

