#!/usr/bin/env python

# This file contains functions for training models. You can run it as follows:
#
#   python train.py -d data -c config -r results -m models
#
# where 'data' is a folder containing the data, 'config' is a file containing the model configs, 'results' is a folder for storing the results and 'model' is a folder to store the models
# verbosity flag.

import argparse
import sys
import yaml
import os

from src.models.model import *
from src.centres.centre import LocalHospital, CentralModelDistributor, ExternalValidationHospital, CentralTrainer


# Parse arguments.
def get_parser():
    description = 'Train model(s).'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    return parser

# Run the code.
def run(args):
    # implement code here:
    central_trainer = CentralTrainer((1000,12), 30)
    for dataset in os.listdir(args.data_folder):
        if dataset.startswith("X") and not dataset.endswith("ptbxl.npy"):
            central_trainer.load_data(os.path.join(args.data_folder,dataset), os.path.join(args.data_folder, dataset.replace("X","y")))
    central_trainer.train_val_split()
    central_trainer.train_to_convergence()
    model = central_trainer.get_model()

    ptb_xl = ExternalValidationHospital("PTB-XL", os.path.join(args.data_folder, "X_data_ptbxl.npy"), os.path.join(args.data_folder, "y_data_ptbxl.npy"))
    ptb_xl.load_model(model)
    fpr, tpr, test_auroc = ptb_xl.predict_test()
    print("AUROC on test data without TFL = ", test_auroc)
    ptb_xl.prepare_for_transfer_learning()
    ptb_xl.train_to_convergence()
    fpr_tfl, tpr_tfl, test_auroc_tfl = ptb_xl.predict_test()
    print("AUROC on test data with TFL = ", test_auroc_tfl)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))