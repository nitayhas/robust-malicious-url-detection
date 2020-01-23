#!/usr/bin/env python

##############################################
# Author: Nitay Hason
# Check all features permutations
##############################################

import itertools
import numpy as np
import pandas as pd
import os

from Models.NeuralNetwork import NeuralNetwork

path               = os.path.dirname(os.path.abspath(__file__))
features_file_name = "../Datasets/features_extractions/median_9_2_(75-25)_vt_include.csv"
features_file      = os.path.join(path, features_file_name)
df_features  = pd.read_csv(features_file)

features = {
    "ld"   : 1, # Length of domain
    "ncc"  : 2, # Number of consecutive characters
    "ed"   : 3, # Entropy of domain
    "nia"  : 4, # Number of IP addresses
    "dgia" : 5, # Distinct geolocations of the IP addresses
    "atv"  : 6, # Average TTL value
    "sdt"  : 7, # Standard deviation of TTL
    "ltd"  : 8, # Life time of domain
    "atd"  : 9, # Active time of domain
    "car"  : 10, # Communication ASNs Rank
    "ccr"  : 11, # Communication Countries Rank
    # "ndr"  : 12, # Number of DNS Records
    "ndc"  : 13, # Number of DNS changes by passive DNS
    # "nsd"  : 14, # Number of Subdomains
    "etsc" : 15, # Expiration Time of SSL Certificate
    # "scv"  : 16 # SSL Certificate is Valid
}
y_column_idx = 17

layers  = [(80,"relu"),(80,"relu"),(80,"leakyrelu"),(1,'sigmoid')]
outputs = {}


for i in range(2,(len(features)+1)):
    combinations = list(itertools.combinations(list(features.keys()),i))
    print("Combinations:")
    print(combinations)
    for combination in combinations:
        print("Current combination:")
        print(combination)
        use_columns = [features[feature_column] for feature_column in combination]
        use_columns.append(y_column_idx)

        new_df = np.array(df_features[df_features.columns[use_columns]].values)
        nn     = NeuralNetwork(dataset=new_df, learning_rate = 0.001, threshold=0.5, training_epochs = 20000, degree=1)
        nn.build()
        nn.train(layers=layers)
        scores = nn.predict()
        del scores["classification_report"]
        scores["confusion_matrix"] = str(scores["confusion_matrix"]).strip()
        outputs[str(combination)]  = scores

out = pd.DataFrame.from_dict(outputs,orient='index')
out.to_csv("features_permutation_output.csv", encoding="utf-8", sep=";")
