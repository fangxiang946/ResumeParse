import process_data
import models.bilsm_crf_model as lsmcrf
import models.cnn_model as cnn
import pickle


EPOCHS = 8
BATCH_SIZE=200

def trainByLstm():
    x_train,y_train,x_val,y_val,embedding_mat = process_data.get_data(cnum=10000,test_size=0.2)

    model = lsmcrf.get_model(x_train,y_train,embedding_mat)

    model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS, validation_data=[x_val,y_val])

    crfmodel_Weights = model.get_weights()

    with open('../ckpt/crfmodel_Weights.pkl', 'wb') as outp:
        pickle.dump(crfmodel_Weights, outp)

    #model.save('../ckpt/crf.h5')
    #model.save_weights('../ckpt/crf_weights.h5')

def trainByCnn():
    x_train, y_train, x_val, y_val, embedding_mat = process_data.get_data(cnum=1000, test_size=0.8)

    model = cnn.get_model(x_train, y_train, embedding_mat)

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=[x_val, y_val])
    model.save('../ckpt/cnn.h5')

if __name__ == '__main__':
    trainByLstm()

