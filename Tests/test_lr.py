#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import time

from Models.LogisticRegression import LogisticRegression


y_column_idx   = 17
features_check = {
	"base": {
		"features": [1,2,3,4,5,6,7,8,9]
	},
	"base_robust": {
		"features": [2,6,8,9]
	},
	"all": {
		"features": [1,2,3,4,5,6,7,8,9,10,11,13,15]
	},
	"novel": {
		"features": [10,11,13,15]
	},
	"hybrid_robust": {
		"features": [2,6,8,9,10,11,13,15]
	}
}

model_name        = "lr"
features_to_check = ["base","base_robust","all","novel","hybrid_robust"]


threshold       = 0.5
degree          = 3
n_splits		= 10
test_size		= 0.25

path               = os.path.dirname(os.path.abspath(__file__))
features_file_name = "../Datasets/features_extractions/median_9_2_(75-25)_vt_include.csv"
features_file      = os.path.join(path, features_file_name)

for features_set in features_to_check:
	print("\n\nChecking features - %s" % (features_set))
	start       = time.time()
	df          = pd.read_csv(features_file)
	use_columns = features_check[features_set]["features"]
	use_columns.append(y_column_idx)

	new_df = df[df.columns[use_columns]]
	new_df = np.array(new_df.values)
	lr = LogisticRegression(new_df, degree=degree, threshold=threshold, kfolds=n_splits, test_size=test_size)
	lr.build()
	lr.train()
	lr.predict()
	end   = time.time()
	print("\nTraining time:")
	print(end - start)
