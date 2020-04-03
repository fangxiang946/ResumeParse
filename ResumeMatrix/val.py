from keras.models import load_model
import process_data
import models.bilsm_crf_model as lsmcrf
import models.cnn_model as cnn
import pickle

BATCH_SIZE=200

def crf_val():
    model = lsmcrf.Init_model()
    with open('../ckpt/crfmodel_Weights.pkl', 'rb') as inp:
        modelWeights = pickle.load(inp)

    model.set_weights(modelWeights)

    x, y = process_data.get_testdata(cnum=1000)

    loss, acc = model.evaluate(x, y, batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))



if __name__ == '__main__':
    crf_val()