import os, sys

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

from trainers.trainer_neural import train, eval

if __name__=='__main__':
    eval_data = "./base/dataset/full_dataset/ifs"
    train_data = "./base/dataset/nr_dataset/N_10" # create this dataset and update path

    base_dir = os.path.dirname(os.path.realpath(__file__))

    # define log dir
    log_path = os.path.join(base_dir, "../log_neural_regression/neural_reg")
    
    num_functions = 10
    train(log_path, train_data, num_functions)

    eval(log_path, eval_data, num_functions)