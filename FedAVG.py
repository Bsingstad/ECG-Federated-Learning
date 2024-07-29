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
    central_unit = CentralModelDistributor((1000,12), 30)
    model = central_unit.get_model()

    st_petersburg = LocalHospital("st_petersburg", os.path.join(args.data_folder, "X_data_stpeter.npy"),os.path.join(args.data_folder, "y_data_stpeter.npy"))
    st_petersburg.load_model(model)

    ptb_diag = LocalHospital("PTB diag", os.path.join(args.data_folder, "X_data_ptbdiag.npy"),os.path.join(args.data_folder, "y_data_ptbdiag.npy"))
    ptb_diag.load_model(model)

    chapman = LocalHospital("chapman", os.path.join(args.data_folder, "X_data_chapman.npy"),os.path.join(args.data_folder, "y_data_chapman.npy"))
    chapman.load_model(model)

    ningbo = LocalHospital("ningbo", os.path.join(args.data_folder, "X_data_ningbo.npy"),os.path.join(args.data_folder, "y_data_ningbo.npy"))
    ningbo.load_model(model)

    georgia = LocalHospital("georgia", os.path.join(args.data_folder, "X_data_georgia.npy"),os.path.join(args.data_folder, "y_data_georgia.npy"))
    georgia.load_model(model)

    chinaphys = LocalHospital("chinaphys", os.path.join(args.data_folder, "X_data_chinaphys.npy"),os.path.join(args.data_folder, "y_data_chinaphys.npy"))
    chinaphys.load_model(model)

    st_petersburg.train_to_convergence()
    stp_fpr, stp_tpr , stp_roc = st_petersburg.predict_test()
    ptb_diag.train_to_convergence()
    ptb_fpr, ptb_tpr , ptb_roc = ptb_diag.predict_test() 
    chapman.train_to_convergence()
    chp_fpr, chp_tpr , chp_roc = chapman.predict_test() 
    ningbo.train_to_convergence()
    ngb_fpr, ngb_tpr , ngb_roc = ningbo.predict_test() 
    georgia.train_to_convergence()
    grg_fpr, grg_tpr , grg_roc = georgia.predict_test() 
    chinaphys.train_to_convergence()
    chn_fpr, chn_tpr , chn_roc = chinaphys.predict_test()

    central_unit.update_weight_list(st_petersburg.get_weights())
    central_unit.update_weight_list(ptb_diag.get_weights())
    central_unit.update_weight_list(chapman.get_weights())
    central_unit.update_weight_list(ningbo.get_weights())
    central_unit.update_weight_list(georgia.get_weights())
    central_unit.update_weight_list(chinaphys.get_weights())
    global_weights = central_unit.return_avg_weights()

    st_petersburg.set_weights(global_weights)
    ptb_diag.set_weights(global_weights)
    chapman.set_weights(global_weights)
    ningbo.set_weights(global_weights)
    georgia.set_weights(global_weights)
    chinaphys.set_weights(global_weights)

    ptb_xl = ExternalValidationHospital("PTB-XL", os.path.join(args.data_folder, "X_data_ptbxl.npy"), os.path.join(args.data_folder, "y_data_ptbxl.npy"))
    ptb_xl.set_weights(global_weights)
    fpr, tpr, test_auroc = ptb_xl.predict_test()
    print("AUROC on test data without TFL = ", test_auroc)
    ptb_xl.prepare_for_transfer_learning()
    ptb_xl.train_to_convergence()
    fpr_tfl, tpr_tfl, test_auroc_tfl = ptb_xl.predict_test()
    print("AUROC on test data with TFL = ", test_auroc_tfl)


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))