import numpy 

from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self._label = label
        
    def __str__(self):
        return str(self._label)
    
    def getLabel(self):
        return self._label

class FeatureVector():
    def __init__(self,indexmax):
        self._feature_vector = numpy.zeros(indexmax) 
        
    def add(self, index, value):
        self._feature_vector[index] = value
        
    def get(self, index):
        return self._feature_vector[index]   
    
    def length(self):
        return len(self._feature_vector)   
    
    def getVector(self):
        return self._feature_vector

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass
    
