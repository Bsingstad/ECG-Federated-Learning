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

    #st_petersburg.train_to_convergence(f"./FedAVG_st_petersburg_history.csv")
    stp_fpr, stp_tpr , stp_roc = st_petersburg.predict_val()
    pd.DataFrame({"fpr":stp_fpr, "tpr": stp_tpr}).to_csv("FedAVG_roc_st_peter_before.csv", index=False)
    ptb_diag.train_to_convergence(f"./FedAVG_ptb_diag_history.csv")
    ptb_fpr, ptb_tpr , ptb_roc = ptb_diag.predict_val() 
    pd.DataFrame({"fpr":ptb_fpr, "tpr": ptb_tpr}).to_csv("FedAVG_roc_ptb_diag_before.csv", index=False)
    chapman.train_to_convergence(f"./FedAVG_chapman_history.csv")
    chp_fpr, chp_tpr , chp_roc = chapman.predict_val() 
    pd.DataFrame({"fpr":chp_fpr, "tpr": chp_tpr}).to_csv("FedAVG_roc_chp_before.csv", index=False)
    ningbo.train_to_convergence(f"./FedAVG_ningbo_history.csv")
    ngb_fpr, ngb_tpr , ngb_roc = ningbo.predict_val()
    pd.DataFrame({"fpr":ngb_fpr, "tpr": ngb_tpr}).to_csv("FedAVG_roc_ngb_before.csv", index=False)
    georgia.train_to_convergence(f"./FedAVG_georgia_history.csv")
    grg_fpr, grg_tpr , grg_roc = georgia.predict_val()
    pd.DataFrame({"fpr":grg_fpr, "tpr": grg_tpr}).to_csv("FedAVG_roc_grg_before.csv", index=False) 
    chinaphys.train_to_convergence(f"./FedAVG_china_history.csv")
    chn_fpr, chn_tpr , chn_roc = chinaphys.predict_val()
    pd.DataFrame({"fpr":chn_fpr, "tpr": chn_tpr}).to_csv("FedAVG_roc_chn_before.csv", index=False)

    central_unit.update_weight_list(st_petersburg.get_weights())
    central_unit.update_weight_list(ptb_diag.get_weights())
    central_unit.update_weight_list(chapman.get_weights())
    central_unit.update_weight_list(ningbo.get_weights())
    central_unit.update_weight_list(georgia.get_weights())
    central_unit.update_weight_list(chinaphys.get_weights())
    global_weights = central_unit.return_avg_weights()

    st_petersburg.set_weights(global_weights)
    stp_fpr, stp_tpr , stp_roc = st_petersburg.predict_val()
    pd.DataFrame({"fpr":stp_fpr, "tpr": stp_tpr}).to_csv("FedAVG_roc_st_peter_after.csv", index=False)
    ptb_diag.set_weights(global_weights)
    ptb_fpr, ptb_tpr , ptb_roc = ptb_diag.predict_val() 
    pd.DataFrame({"fpr":ptb_fpr, "tpr": ptb_tpr}).to_csv("FedAVG_roc_ptb_diag_after.csv", index=False)
    chapman.set_weights(global_weights)
    chp_fpr, chp_tpr , chp_roc = chapman.predict_val() 
    pd.DataFrame({"fpr":chp_fpr, "tpr": chp_tpr}).to_csv("FedAVG_roc_chp_after.csv", index=False)
    ningbo.set_weights(global_weights)
    ngb_fpr, ngb_tpr , ngb_roc = ningbo.predict_val()
    pd.DataFrame({"fpr":ngb_fpr, "tpr": ngb_tpr}).to_csv("FedAVG_roc_ngb_after.csv", index=False)
    georgia.set_weights(global_weights)
    grg_fpr, grg_tpr , grg_roc = georgia.predict_val()
    pd.DataFrame({"fpr":grg_fpr, "tpr": grg_tpr}).to_csv("FedAVG_roc_grg_after.csv", index=False) 
    chinaphys.set_weights(global_weights)
    chn_fpr, chn_tpr , chn_roc = chinaphys.predict_val()
    pd.DataFrame({"fpr":chn_fpr, "tpr": chn_tpr}).to_csv("FedAVG_roc_chn_after.csv", index=False)

    ptb_xl = ExternalValidationHospital("PTB-XL", os.path.join(args.data_folder, "X_data_ptbxl.npy"), os.path.join(args.data_folder, "y_data_ptbxl.npy"))
    ptb_xl.load_model(model)
    ptb_xl.set_weights(global_weights)
    fpr, tpr, test_auroc = ptb_xl.predict()
    print("AUROC on PTB-XL = ", test_auroc)

    pd.DataFrame({"fpr":fpr, "tpr": tpr}).to_csv("FedAVG_roc_PTBXL.csv", index=False)


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))