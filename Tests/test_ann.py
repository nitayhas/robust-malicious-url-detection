#!/usr/bin/env python

import itertools
import numpy as np
import pandas as pd
import os
import time
import random

from Models.NeuralNetwork import NeuralNetwork

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


features_check = {
	"base": {
		"features"	   : [1,2,3,4,5,6,7,8,9],
		"y_column_idx" : 10,
		"feature_file" : "../Datasets/features_extractions/base_(all).csv"
	},
	"base_robust": {
		"features": [2,6,8,9],
		"y_column_idx" : 10,
		"feature_file" : "../Datasets/features_extractions/base_(all).csv"
	},
	"all": {
		"features": [1,2,3,4,5,6,7,8,9,10,11,13,15],
		"y_column_idx" : 17,
		"feature_file" : "../Datasets/features_extractions/median_9_2_(25-75)_vt_include.csv"
	},
	"novel": {
		"features": [10,11,13,15],
		"y_column_idx" : 17,
		"feature_file" : "../Datasets/features_extractions/median_9_2_(25-75)_vt_include.csv"
	},
	"hybrid_robust": {
		"features": [2,6,8,9,10,11,13,15],
		"y_column_idx" : 17,
		"feature_file" : "../Datasets/features_extractions/median_9_2_(25-75)_vt_include.csv"
	}
}
features_to_check = ["base"]


threshold       = 0.5
learning_rate   = 0.001
training_epochs = 20000
degree          = 2
n_splits		= 10
test_size		= 0.25

path               = os.path.dirname(os.path.abspath(__file__))
# features_file_name = "../Datasets/features_extractions/median_9_2_(25-75)_vt_include.csv"

for features_set in features_to_check:
	print("\n\nChecking features - %s" % (features_set))
	features_file = os.path.join(path, features_check[features_set]["feature_file"])
	y_column_idx  = features_check[features_set]["y_column_idx"]
	start         = time.time()
	df            = pd.read_csv(features_file)
	######## Append artificial data by number of consecutive characters feature ########
	if 2 in features_check[features_set]["features"]:
		mal         = df[df[df.columns[y_column_idx]]==1].sample(500).copy()
		mal["2"]    = mal["2"].apply(lambda x:x*random.randint(3,9))
		df = df.append(mal, ignore_index=True)
	######################################## END #######################################
	use_columns   = features_check[features_set]["features"]
	use_columns.append(y_column_idx)

	new_df = df[df.columns[use_columns]]
	new_df = np.array(new_df.values)
	nn     = NeuralNetwork(dataset=new_df, learning_rate=learning_rate, threshold=threshold, kfolds=n_splits, training_epochs=training_epochs, degree=degree)
	nn.build()
	nn.train(verbose=1)
	scores = nn.predict()
	nn.save_model("ann_model_base_t")

	end   = time.time()
	print("\nTraining time:")
	print(end - start)
