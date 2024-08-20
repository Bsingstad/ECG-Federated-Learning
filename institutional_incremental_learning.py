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
    central_unit = CentralModelDistributor((1000,12), 30)
    model = central_unit.get_model()

    st_petersburg = LocalHospital("st_petersburg", os.path.join(args.data_folder, "X_data_stpeter.npy"),os.path.join(args.data_folder, "y_data_stpeter.npy"))
    st_petersburg.load_model(model)
    st_petersburg.train_to_convergence("./institutional_incr_learning_st_petersburg.csv")
    temp_weights = st_petersburg.get_weights()
    stp_fpr, stp_tpr , stp_roc = st_petersburg.predict_val()
    print("AUROC on st.petersburg diag = ", stp_roc)
    
    pd.DataFrame({"fpr":stp_fpr, "tpr": stp_tpr}).to_csv("institutional_incr_learning_roc_st_peter.csv")


    ptb_diag = LocalHospital("PTB diag", os.path.join(args.data_folder, "X_data_ptbdiag.npy"),os.path.join(args.data_folder, "y_data_ptbdiag.npy"))
    ptb_diag.load_model(model)
    ptb_diag.set_weights(temp_weights)
    ptb_diag.train_to_convergence("./institutional_incr_learning_ptb_diag.csv")
    temp_weights = ptb_diag.get_weights()
    ptb_fpr, ptb_tpr , ptb_roc = ptb_diag.predict_val() 
    print("AUROC on ptb diag = ", ptb_roc)
    
    pd.DataFrame({"fpr":ptb_fpr, "tpr": ptb_tpr}).to_csv("institutional_incr_learning_roc_ptb_diag.csv")


    chapman = LocalHospital("chapman", os.path.join(args.data_folder, "X_data_chapman.npy"),os.path.join(args.data_folder, "y_data_chapman.npy"))
    chapman.load_model(model)
    chapman.set_weights(temp_weights)
    chapman.train_to_convergence("./institutional_incr_learning_chapman.csv")
    temp_weights = chapman.get_weights()
    chp_fpr, chp_tpr , chp_roc = chapman.predict_val() 
    print("AUROC on chapman = ", chp_roc)
    
    pd.DataFrame({"fpr":chp_fpr, "tpr": chp_tpr}).to_csv("institutional_incr_learning_roc_chapman.csv")


    ningbo = LocalHospital("ningbo", os.path.join(args.data_folder, "X_data_ningbo.npy"),os.path.join(args.data_folder, "y_data_ningbo.npy"))
    ningbo.load_model(model)
    ningbo.set_weights(temp_weights)
    ningbo.train_to_convergence("./institutional_incr_learning_ningbo.csv")
    temp_weights = ningbo.get_weights()
    ngb_fpr, ngb_tpr , ngb_roc = ningbo.predict_val() 
    print("AUROC on ningbo = ", ngb_roc)
    
    pd.DataFrame({"fpr":ngb_fpr, "tpr": ngb_tpr}).to_csv("institutional_incr_learning_roc_ningbo.csv")


    georgia = LocalHospital("georgia", os.path.join(args.data_folder, "X_data_georgia.npy"),os.path.join(args.data_folder, "y_data_georgia.npy"))
    georgia.load_model(model)
    georgia.set_weights(temp_weights)
    georgia.train_to_convergence("./institutional_incr_learning_georgia.csv")
    temp_weights = georgia.get_weights()
    grg_fpr, grg_tpr , grg_roc = georgia.predict_val()
    print("AUROC on georgia = ", grg_roc)
    
    pd.DataFrame({"fpr":grg_fpr, "tpr": grg_tpr}).to_csv("institutional_incr_learning_roc_georgia.csv")


    chinaphys = LocalHospital("chinaphys", os.path.join(args.data_folder, "X_data_chinaphys.npy"),os.path.join(args.data_folder, "y_data_chinaphys.npy"))
    chinaphys.load_model(model)
    chinaphys.set_weights(temp_weights)
    chinaphys.train_to_convergence("./institutional_incr_learning_chinaphys.csv")
    final_weights = chinaphys.get_weights()
    chn_fpr, chn_tpr , chn_roc = chinaphys.predict_val()
    print("AUROC on chinaphys = ", chn_roc)
    
    pd.DataFrame({"fpr":chn_fpr, "tpr": chn_tpr}).to_csv("institutional_incr_learning_roc_chinaphys.csv")

    ptb_xl = ExternalValidationHospital("PTB-XL", os.path.join(args.data_folder, "X_data_ptbxl.npy"), os.path.join(args.data_folder, "y_data_ptbxl.npy"))
    ptb_xl.set_weights(final_weights)
    fpr, tpr, test_auroc = ptb_xl.predict()
    print("AUROC on PTB-XL = ", test_auroc)

    pd.DataFrame({"fpr":fpr, "tpr": tpr}).to_csv("institutional_incr_learning_roc_PTB.csv")


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))