import pickle
import sklearn

# load the model from disk
filename = 'test_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


def classify(text):
    return loaded_model.predict([text])[0]
