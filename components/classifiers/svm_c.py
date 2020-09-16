from components.classifier import Classifier
from sklearn import svm

class SVM_C(Classifier):
    
    def __init__(self, params):
        super().__init__()
        self.__params = params
        self.__define_model()

    def __define_model(self):
        params = self.__params
        clf = svm.SVC(kernel=params['Kernel'], C=params['C'], degree=params['Degree'], gamma=params['Gamma'])
        self.__model = clf
        
    def train_model(self, X, y):
        self.__model.fit(X, y)

    def predict(self, y):
        return self.__model.predict(y)
