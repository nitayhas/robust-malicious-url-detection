#!/usr/bin/env python

import itertools
import numpy as np
import pandas as pd
import os
import time
import random

from Models.ELM import ELMClassifier

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
from sklearn.model_selection import KFold


features_check = {
	"base": {
		"features"     : [1,2,3,4,5,6,7,8,9],
		"C"            : 0.001,
		"n_hidden"     : 50,
		"y_column_idx" : 10,
		"feature_file" : "../Datasets/features_extractions/base_(all).csv"
	},
	"base_robust": {
		"features"     : [2,6,8,9],
		"C"            : 0.001,
		"n_hidden"     : 10,
		"y_column_idx" : 10,
		"feature_file" : "../Datasets/features_extractions/base_(all).csv"
	},
	"all": {
		"features"     : [1,2,3,4,5,6,7,8,9,10,11,13,15],
		"C"            : 50,
		"n_hidden"     : 150,
		"y_column_idx" : 17,
		"feature_file" : "../Datasets/features_extractions/median_9_2_(25-75)_vt_include.csv"
	},
	"novel": {
		"features"     : [10,11,13,15],
		"C"            : 0.004,
		"n_hidden"     : 50,
		"y_column_idx" : 17,
		"feature_file" : "../Datasets/features_extractions/median_9_2_(25-75)_vt_include.csv"
	},
	"hybrid_robust": {
		"features"     : [2,6,8,9,10,11,13,15],
		"C"            : 0.01,
		"n_hidden"     : 100,
		"y_column_idx" : 17,
		"feature_file" : "../Datasets/features_extractions/median_9_2_(25-75)_vt_include.csv"
	}
}


model_name        = "elm"
features_to_check = ["base","base_robust","all","novel","hybrid_robust"]

threshold       = 0.5
learning_rate   = 0.001
n_splits		= 10
test_size		= 0.25

path               = os.path.dirname(os.path.abspath(__file__))
features_file_name = "../Datasets/features_extractions/median_9_2_(75-25)_vt_include.csv"
features_file      = os.path.join(path, features_file_name)


for features_set in features_to_check:
	print("\n\nChecking features - %s" % (features_set))
	features_file = os.path.join(path, features_check[features_set]["feature_file"])
	y_column_idx  = features_check[features_set]["y_column_idx"]
	n_hidden      = features_check[features_set]["n_hidden"]
	train         = pd.read_csv(features_file)
	######## Append artificial data by number of consecutive characters feature ########
	if 2 in features_check[features_set]["features"]:
		mal         = train[train[train.columns[y_column_idx]]==1].sample(500).copy()
		mal["2"]    = mal["2"].apply(lambda x:x*random.randint(3,9))
		train = train.append(mal, ignore_index=True)
	######################################## END #######################################
	use_columns   = features_check[features_set]["features"]
	use_columns.append(y_column_idx)

	train = train[train.columns[use_columns]]

	use_dataset       = train.copy()
	use_dataset       = np.asfarray(use_dataset.values,np.dtype('Float64'))
	# Normlize the dataset
	scaler = MinMaxScaler().fit(use_dataset[:, :-1])
	dataset_norm = scaler.transform(use_dataset[:, :-1])

	# Split features and labels
	X, y = use_dataset, np.transpose([use_dataset[:, -1]])

	indices = np.arange(y.shape[0])
	X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, stratify=y, test_size=test_size, random_state=42)


	kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
	kf.get_n_splits(X_train)

	start = time.time()
	elm   = ELMClassifier(n_hidden=n_hidden, C= features_check[features_set]["C"], activation='relu') # activation = ['relu','tanh','logistic']

	accuracies = []
	losses     = []
	f1s        = []
	precisions = []
	recalls    = []
	auc_scores = []

	for train_index, test_index in kf.split(idx_train):
		X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
		y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

		elm.fit(X_train_fold,y_train_fold)

		y_pred       = elm.predict_proba(X_test_fold)[:, 1]
		yhat_classes = y_pred.copy()
		yhat_classes[yhat_classes>=threshold] = np.float64(1)
		yhat_classes[yhat_classes<threshold]  = np.float64(0)

		accuracy  = accuracy_score(y_test_fold, yhat_classes)
		loss      = log_loss(y_test_fold, yhat_classes)
		f1        = f1_score(y_test_fold, yhat_classes)
		precision = precision_score(y_test_fold, yhat_classes)
		recall    = recall_score(y_test_fold, yhat_classes)
		auc_score = roc_auc_score(y_test_fold, y_pred)

		accuracies.append(accuracy)
		losses.append(loss)
		f1s.append(f1)
		precisions.append(precision)
		recalls.append(recall)
		auc_scores.append(auc_score)

	end   = time.time()
	print('Accuracy: %f' % np.array(accuracies).mean())
	print('Precision: %f' % np.array(precisions).mean())
	print('Recall: %f' % np.array(recalls).mean())
	print('F1 score: %f' % np.array(f1s).mean())
	print('Loss: %f' % np.array(losses).mean())
	print('ROC AUC: %f' % np.array(auc_scores).mean())

	print("\nTraining time:")
	print(end - start)
