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
    central_unit = CentralModelDistributor((1000,12), 30)
    model = central_unit.get_model()
    NUM_ROUNDS = 10

    
    ptb_xl = ExternalValidationHospital("PTB-XL", os.path.join(args.data_folder, "X_data_ptbxl.npy"), os.path.join(args.data_folder, "y_data_ptbxl.npy"))
    ptb_xl.load_model(model)

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

    stp_roc_list = []
    ptb_roc_list = []
    chp_roc_list = []
    ngb_roc_list = []
    grg_roc_list = []
    chn_roc_list = []
    ptb_xl_list = []

    for i in range(NUM_ROUNDS):

        #st_petersburg.train_to_convergence()
        ptb_diag.train_to_convergence(f"./repeated_FedAVG_ptb_diag_history_round_{i}.csv")
        chapman.train_to_convergence(f"./repeated_FedAVG_chapman_history_round_{i}.csv")
        ningbo.train_to_convergence(f"./repeated_FedAVG_ningbo_history_round_{i}.csv")
        georgia.train_to_convergence(f"./repeated_FedAVG_georgia_history_round_{i}.csv")
        chinaphys.train_to_convergence(f"./repeated_FedAVG_china_history_round_{i}.csv")

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
        ptb_xl.set_weights(global_weights)

        
        stp_fpr, stp_tpr , stp_roc = st_petersburg.predict_val()
        ptb_fpr, ptb_tpr , ptb_roc = ptb_diag.predict_val() 
        chp_fpr, chp_tpr , chp_roc = chapman.predict_val() 
        ngb_fpr, ngb_tpr , ngb_roc = ningbo.predict_val() 
        grg_fpr, grg_tpr , grg_roc = georgia.predict_val() 
        chn_fpr, chn_tpr , chn_roc = chinaphys.predict_val()
        fpr_ptbxl, tpr_ptbxl, ptbxl_roc = ptb_xl.predict()

        stp_roc_list.append(stp_roc)
        ptb_roc_list.append(ptb_roc)
        chp_roc_list.append(chp_roc)
        ngb_roc_list.append(ngb_roc)
        grg_roc_list.append(grg_roc)
        chn_roc_list.append(chn_roc)
        ptb_xl_list.append(ptbxl_roc)

    ptb_xl.set_weights(global_weights)
    fpr, tpr, test_auroc = ptb_xl.predict()
    print("AUROC on PTB-XL = ", test_auroc)

    pd.DataFrame({"fpr":fpr, "tpr": tpr}).to_csv("repeated_FedAVG_roc_PTBXL.csv", index=False)

    pd.DataFrame({"st_petersburg":stp_roc_list, "ptb_diag": ptb_roc_list, "chapman":chp_roc_list,
                  "ningbo":ngb_roc_list, "georgia": grg_roc_list, "china":chn_roc_list, "ptbxl":ptb_xl_list}).to_csv("training_history_repeated_FedAVG.csv", index=False)



if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))