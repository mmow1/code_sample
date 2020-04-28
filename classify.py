import os
import argparse
import sys
import pickle
import numpy 

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

def load_data(filename):
    instances = []
    global indexmax        
    indexmax = 1
    with open(filename) as reader:
        
        for line in reader:  
            split_line = line.split(" ")
            
            for item in split_line[1:]:
                index = int(item.split(":")[0])
                if index > indexmax:
                    indexmax = index
        indexmax = indexmax + 1
    if pastindexmax != -1:
        indexmax = pastindexmax
    with open(filename) as reader:
               
        for line in reader:
            if len(line.strip()) == 0:
                continue
            
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")

            label = ClassificationLabel(int_label)
            
            
            feature_vector = FeatureVector(indexmax)
               
            for item in split_line[1:]:
                try:
                    index = int(item.split(":")[0])
                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                
                if value != 0.0:
                    if index <= indexmax:
                        feature_vector.add(index, value)

            instance = Instance(feature_vector, label)
            instances.append(instance)
    return instances

def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True, help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")
    parser.add_argument("--online-learning-rate", type=float, help="The learning rate for perceptron", default=1.0)
    parser.add_argument("--online-training-iterations", type=int, help="The number of training iterations for online methods.", default=5)

    
    args = parser.parse_args()
    check_args(args)

    return args

def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--algorithm should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")

class myPredictor():
    def __init__(self,instances,n,I):
        self.w = numpy.zeros(indexmax)
        self.n = n
        self.I = I
        self.wtot = numpy.zeros(indexmax)
        
    def train(self, instances, algorithm):
        if algorithm == "mc_perceptron":
            self.K = 0;
            for i in range(len(instances)):
                x = instances[i]
                if x._label.getLabel() > self.K:
                    self.K = x._label.getLabel()          
            self.w = numpy.zeros([indexmax,self.K])

            for k in range(self.I):
                for i in range(len(instances)):
                    x = instances[i]
                    yhat = self.predict(x)    

                    if yhat != x._label.getLabel():
                        x2 = x._feature_vector.getVector()
                        x2 = x2[0:indexmax]
                        y = x._label.getLabel()
                        self.w[:,yhat-1] = self.w[:,yhat-1] - x2
                        self.w[:,y-1] = self.w[:,y-1] + x2    
            
    def predict(self, x):
        x2 = x._feature_vector.getVector()
        x2 = x2[0:indexmax]
        yhat = 0
        fprev = 0
        for i in range(self.K):
            f = numpy.dot(self.w[:,i],x2)
            if f > fprev:
                yhat = i
                fprev = f
        return yhat+1
    
    def getLength(self):
        return len(self.w)
    
def train(instances, algorithm, n, I):
    p = myPredictor(instances,n,I)
    p.train(instances,algorithm)
    return p

def write_predictions(predictor, instances, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for instance in instances:
                label = predictor.predict(instance)
        
                writer.write(str(label))
                writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)

def main():
    args = get_args()
    global pastindexmax
    if args.mode.lower() == "train":
        # Load the training data.
        pastindexmax = -1
        instances = load_data(args.data)
        pastindexmax = indexmax

        # Train the model.
        predictor = train(instances, args.algorithm, args.online_learning_rate, args.online_training_iterations)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
        
    elif args.mode.lower() == "test":

        
        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
        
        pastindexmax = predictor.getLength()
        # Load the test data.
        instances = load_data(args.data)
        
        write_predictions(predictor, instances, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()

