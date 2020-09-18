import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Model
from sklearn.model_selection import StratifiedKFold, train_test_split
from components.preprocessor import Preprocessor
from components.classifier import Classifier
from keras import backend as K

class CNN_SVM(Preprocessor, Classifier):

    
    def __init__(self, params, num_classes, shape):
        self.__data_loaded = False
        self.__model_trained = False
        self.__params = params
        self.__num_classes = num_classes
        self.__shape = shape
        self.__define_model()

    def __define_model(self):
        params = self.__params
        shape = self.__shape
        if(params['RGB'] == False):
            shape = (shape[0],shape[1],1)

        model = Sequential()
        model.add(Input(shape=shape))
        for i in range(params['ConvCount']):
            cur_filt_size = params['ConvFilterSizes'][i]
            cur_filt_count = params['ConvFilterCount'][i]
            model.add(Conv2D(cur_filt_count, kernel_size=(cur_filt_size, cur_filt_size), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        firstDenseLayer = params['FirstDenseLayer']
        finalFeatVec = params['FinalFeatVec']

        model.add(Dense(firstDenseLayer, activation='relu'))
        if(finalFeatVec != None):
            model.add(Dense(finalFeatVec, activation='relu'))
        model.add(Dense(self.__num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.__model = model

    def reinit(self):
        K.clear_session()
        self.__define_model()
        


    def train_model(self, X, y):
        if(len(X.shape) < 4):
            X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
        y_categ = to_categorical(y, self.__num_classes)
        res = self.__model.train_on_batch(X, y_categ)
        #print(res)
        return res

    def predict(self, X):
        if(len(X.shape) < 4):
            X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)

        return np.argmax(self.__model.predict(X), axis=-1)
        
    def get_out_data(self, new_data):
        old_model = self.__model
        sh = new_data.shape
        if(len(new_data.shape) < 4):
            new_data = new_data.reshape(sh[0],sh[1],sh[2],1)

        layer_count = len(old_model.layers)
        new_model = Model(inputs=old_model.input, outputs=old_model.layers[layer_count-2].output)
        new_features = new_model.predict(new_data)
        return new_features

