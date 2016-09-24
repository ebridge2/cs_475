import os
import argparse
import sys
import pickle

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
from perceptron import Perceptron, Weighted_Perceptron

def load_data(filename):
    instances = []
    nfeatures = 0 # initialize the number of features of our data
    with open(filename) as reader:
        for line in reader:
            if len(line.strip()) == 0:
                continue
            
            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")

            label = ClassificationLabel(int_label)
            feature_vector = FeatureVector()
            
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
                    feature_vector.add(index, value)
            instance = Instance(feature_vector, label)
            instances.append(instance)
            # since we have a sparse vector, we might not necessarily see the total number of features
            # in a given instance, so track the highest feature number we see
            # ie, some examples might have features up to 75, but another might only have 74 since the 
            # 75th for that example was 0
            nfeatures = max(feature_vector.max_feature(), nfeatures)
    return tuple((instances, nfeatures))


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")
    parser.add_argument("--online-learning-rate", type=float, help="The learning rate for perceptron",
                        default=1.0)
    parser.add_argument("--online-training-iterations", type=int,
                        help="The number of training iterations for online methods.", default=5)    
    
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

def train(instances, algorithm, nfeatures, rate, iterations):
    # if the user explicitly requests a weighted model
    if (algorithm == "averaged_perceptron"):
        predictor = Weighted_Perceptron(nfeatures, rate, iterations)
    elif (algorithm == "perceptron"): # use a simple perceptron model
        predictor = Perceptron(nfeatures, rate, iterations)
    # train it on the data
    else:
        raise ValueError("You did not pass a relevant algorithm name." +
                         "Options are 'averaged_perceptron' and 'perceptron'.")
    predictor.train(instances)
    return predictor # return it back


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

    if args.mode.lower() == "train":
        # Load the training data.
        (instances, nfeatures) = load_data(args.data)

        # Train the model.
        predictor = train(instances, args.algorithm, nfeatures, args.online_learning_rate,
                          args.online_training_iterations)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        (instances, nfeatures) = load_data(args.data)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        write_predictions(predictor, instances, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()

