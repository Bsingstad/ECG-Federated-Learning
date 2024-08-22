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
    NUM_ROUNDS = 10

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

    ptb_xl = ExternalValidationHospital("PTB-XL", os.path.join(args.data_folder, "X_data_ptbxl.npy"), os.path.join(args.data_folder, "y_data_ptbxl.npy"))
    ptb_xl.load_model(model)

    stp_roc_list = []
    ptb_roc_list = []
    chp_roc_list = []
    ngb_roc_list = []
    grg_roc_list = []
    chn_roc_list = []
    test_auroc_list = []
    
    for i in range(NUM_ROUNDS):
        #st_petersburg.train_to_convergence(f"./cyclic_institutional_incr_learning_st_peter_{i}.csv")
        temp_weights = st_petersburg.get_weights()
        stp_fpr, stp_tpr , stp_roc = st_petersburg.predict_val()
        print(f"AUROC on st.petersburg diag round {i}= ", stp_roc)
        stp_roc_list.append(stp_roc)
        pd.DataFrame({"fpr":stp_fpr, "tpr": stp_tpr}).to_csv(f"cycl_institutional_incr_learning_roc_st_peter_round_{i}.csv")

        ptb_diag.set_weights(temp_weights)
        ptb_diag.train_to_convergence(f"./cyclic_institutional_incr_learning_ptb_diag_{i}.csv")
        temp_weights = ptb_diag.get_weights()
        ptb_fpr, ptb_tpr , ptb_roc = ptb_diag.predict_val() 
        print(f"AUROC on ptb diag round {i}= ", ptb_roc)
        ptb_roc_list.append(ptb_roc)
        pd.DataFrame({"fpr":ptb_fpr, "tpr": ptb_tpr}).to_csv(f"cycl_institutional_incr_learning_roc_ptb_diag_round_{i}.csv")

        chapman.set_weights(temp_weights)
        chapman.train_to_convergence(f"./cyclic_institutional_incr_learning_chapman_{i}.csv")
        temp_weights = chapman.get_weights()
        chp_fpr, chp_tpr , chp_roc = chapman.predict_val() 
        print(f"AUROC on chapman round {i} = ", chp_roc)
        chp_roc_list.append(chp_roc)
        pd.DataFrame({"fpr":chp_fpr, "tpr": chp_tpr}).to_csv(f"cycl_institutional_incr_learning_roc_chapman_round_{i}.csv")

        ningbo.set_weights(temp_weights)
        ningbo.train_to_convergence(f"./cyclic_institutional_incr_learning_ningbo_{i}.csv")
        temp_weights = ningbo.get_weights()
        ngb_fpr, ngb_tpr , ngb_roc = ningbo.predict_val() 
        print(f"AUROC on ningbo round {i} = ", ngb_roc)
        ngb_roc_list.append(ngb_roc)
        pd.DataFrame({"fpr":ngb_fpr, "tpr": ngb_tpr}).to_csv(f"cycl_institutional_incr_learning_roc_ningbo_round_{i}.csv")


        georgia.set_weights(temp_weights)
        georgia.train_to_convergence(f"./cyclic_institutional_incr_learning_georgia_{i}.csv")
        temp_weights = georgia.get_weights()
        grg_fpr, grg_tpr , grg_roc = georgia.predict_val()
        print(f"AUROC on georgia round {i}= ", grg_roc)
        grg_roc_list.append(grg_roc)
        pd.DataFrame({"fpr":grg_fpr, "tpr": grg_tpr}).to_csv(f"cycl_institutional_incr_learning_roc_georgia_round_{i}csv")


        chinaphys.set_weights(temp_weights)
        chinaphys.train_to_convergence(f"./cyclic_institutional_incr_learning_chinaphys_{i}.csv")
        temp_weights = chinaphys.get_weights()
        chn_fpr, chn_tpr , chn_roc = chinaphys.predict_val()
        print(f"AUROC on chinaphys round {i} = ", chn_roc)
        chn_roc_list.append(chn_roc)
        pd.DataFrame({"fpr":chn_fpr, "tpr": chn_tpr}).to_csv(f"cycl_institutional_incr_learning_roc_chinaphys_round_{i}.csv")


        ptb_xl.set_weights(temp_weights)
        fpr, tpr, test_auroc = ptb_xl.predict()
        print(f"AUROC on PTB-XL round {i} = ", test_auroc)
        test_auroc_list.append(test_auroc)
        pd.DataFrame({"fpr":fpr, "tpr": tpr}).to_csv("cycl_institutional_incr_learning_roc_PTB_round_{i}.csv")


    pd.DataFrame({"st_petersburg":stp_roc_list, "ptb_diag": ptb_roc_list, "chapman":chp_roc_list,
                  "ningbo":ngb_roc_list, "georgia": grg_roc_list, "china":chn_roc_list, "ptbxl":test_auroc_list}).to_csv("training_history_cycl_inst_learn.csv", index=False)




if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))