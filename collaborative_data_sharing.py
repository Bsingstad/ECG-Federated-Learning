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
import pandas as pd

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

    st_petersburg = LocalHospital("st_petersburg", os.path.join(args.data_folder, "X_data_stpeter.npy"),os.path.join(args.data_folder, "y_data_stpeter.npy"))
    ptb_diag = LocalHospital("PTB diag", os.path.join(args.data_folder, "X_data_ptbdiag.npy"),os.path.join(args.data_folder, "y_data_ptbdiag.npy"))
    chapman = LocalHospital("chapman", os.path.join(args.data_folder, "X_data_chapman.npy"),os.path.join(args.data_folder, "y_data_chapman.npy"))
    ningbo = LocalHospital("ningbo", os.path.join(args.data_folder, "X_data_ningbo.npy"),os.path.join(args.data_folder, "y_data_ningbo.npy"))
    georgia = LocalHospital("georgia", os.path.join(args.data_folder, "X_data_georgia.npy"),os.path.join(args.data_folder, "y_data_georgia.npy"))
    chinaphys = LocalHospital("chinaphys", os.path.join(args.data_folder, "X_data_chinaphys.npy"),os.path.join(args.data_folder, "y_data_chinaphys.npy"))
    
    central_trainer.load_train_data_from_centre(*st_petersburg.get_data(split="train"))
    central_trainer.load_val_data_from_centre(*st_petersburg.get_data(split="val"))

    central_trainer.load_train_data_from_centre(*ptb_diag.get_data(split="train"))
    central_trainer.load_val_data_from_centre(*ptb_diag.get_data(split="val"))

    central_trainer.load_train_data_from_centre(*chapman.get_data(split="train"))
    central_trainer.load_val_data_from_centre(*chapman.get_data(split="val"))

    central_trainer.load_train_data_from_centre(*ningbo.get_data(split="train"))
    central_trainer.load_val_data_from_centre(*ningbo.get_data(split="val"))

    central_trainer.load_train_data_from_centre(*georgia.get_data(split="train"))
    central_trainer.load_val_data_from_centre(*georgia.get_data(split="val"))

    central_trainer.load_train_data_from_centre(*chinaphys.get_data(split="train"))
    central_trainer.load_val_data_from_centre(*chinaphys.get_data(split="val"))

    central_trainer.train_to_convergence("./collaborative_data_sharing.csv")
    model = central_trainer.get_model()

    ptb_xl = ExternalValidationHospital("PTB-XL", os.path.join(args.data_folder, "X_data_ptbxl.npy"), os.path.join(args.data_folder, "y_data_ptbxl.npy"))
    ptb_xl.load_model(model)
    fpr, tpr, test_auroc = ptb_xl.predict()
    print("AUROC on PTB-XL = ", test_auroc)

    pd.DataFrame({"fpr":fpr, "tpr": tpr}).to_csv("collaborative_data_sharing_roc.csv")


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))