#!/usr/bin/env python

##############################################
# Author: Nitay Hason
# Check feature robustness
##############################################

import itertools
import numpy as np
import pandas as pd
import sys, os
import time
import random

from Tools.FeaturesExtraction import FeaturesExtraction
from Models.NeuralNetwork import NeuralNetwork
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import math

def to_percent(y, position):
	s = str(100 * round(y, 2))
	if plt.rcParams['text.usetex'] is True:
		return s + r'$\%$'
	else:
		return s + '%'


features = {
	1 : "(a) Length of domain",
	2 : "(b) * Number of consecutive characters",
	3 : "(c) Entropy of domain",
	4 : "(d) Number of IP addresses",
	5 : "(e) Distinct geolocations of the IP addresses",
	6 : "(f) * Mean TTL value",
	7 : "(g) Standard deviation of TTL",
	8 : "(h) * Life time of domain",
	9 : "(i) * Active time of domain",
	10: "(j) Communication Countries Rank",
	11: "(k) Communication ASNs Rank",
	12: "Number of DNS Records",
	13: "(l) Number of DNS changes by passive DNS",
	14: "Number of Subdomains",
	15: "(m) Expiration Time of SSL Certificate",
	16: "SSL Certificate is Valid"
}

# y_column_idx   = 17 # Label column index
features_check = {  # Feautres sets, include manipulation options
	"base": {
		"features"        : [1,2,3,4,5,6,7,8,9],
		"y_column_idx"    : 10,
		"change_features" : [[0,list(np.arange(6,30,1))],[1,list(np.array(np.arange(1.,10.,1.))[::-1])],[2,list(np.arange(2.0,15.25,0.25))],[3,list(np.arange(1.,31.0,1.0))],[4,list(np.arange(1.,11.,1))],[5,list(np.arange(0,60001,10000))],[6,list(np.arange(0,60001,10000))],[7,list(np.arange(1,21,1))],[8,list(np.arange(0,21,1))]],
		"feature_file"    : "../Datasets/features_extractions/base_(all).csv"
	},
	"base_robust": {
		"features"        : [2,6,8,9],
		"y_column_idx"    : 10,
		"change_features" : [[0,list(np.array(np.arange(1.,10.,1.))[::-1])],[1,list(np.arange(1,60001,10000))],[2,list(np.arange(1,21,1))],[3,list(np.arange(0,21,1))]],
		"feature_file"    : "../Datasets/features_extractions/base_(all).csv"
	},
	"all": {
		"features"        : [1,2,3,4,5,6,7,8,9,10,11,13,15],
		"y_column_idx"    : 17,
		"change_features" : [[0,list(np.arange(6,30,1))],[1,list(np.array(np.arange(1.,10.,1.))[::-1])],[2,list(np.arange(2.0,15.25,0.25))],[3,list(np.arange(1.,31.0,1.0))],[4,list(np.arange(1.,11.,1))],[5,list(np.arange(0,60001,10000))],[6,list(np.arange(0,60001,10000))],[7,list(np.arange(1,21,1))],[8,list(np.arange(0,21,1))],[9,list(np.arange(0,4051,90))],[10,list(np.arange(0,720001,3000))],[11,list(np.arange(0,100,2))],[12,list(np.arange(0,15810000,100000))]],
		"feature_file"    : "../Datasets/features_extractions/median_9_2_(75-25)_vt_include.csv"
	},
	"novel": {
		"features"        : [10,11,13,15],
		"y_column_idx"    : 17,
		"change_features" : [[0,list(np.arange(0,4051,90))],[1,list(np.arange(0,720001,3000))],[2,list(np.arange(0,100,2))],[3,list(np.arange(0,15810000,100000))]],
		"feature_file"    : "../Datasets/features_extractions/median_9_2_(75-25)_vt_include.csv"
	},
	"robust": {
		"features"        : [2,6,8,9,10,11,13,15],
		"y_column_idx"    : 17,
		"change_features" : [[0,list(np.array(np.arange(1.,10.,1.))[::-1])],[1,list(np.arange(0,60001,10000))],[2,list(np.arange(1,21,1))],[3,list(np.arange(0,21,1))],[4,list(np.arange(0,4051,90))],[5,list(np.arange(0,720001,3000))],[6,list(np.arange(0,100,2))],[7,list(np.arange(0,15810000,100000))]],
		"feature_file"    : "../Datasets/features_extractions/median_9_2_(75-25)_vt_include.csv"
	}
}

# Malicious sample example
# ['http://paypal.co.uk.userpph0v45y2pp.settingsppup.com/id1/?eml=&amp;cmd=form_submit&amp;dispatch=34xsd45423d1zmw241234zxadvzvh24af23d60', 16,2,0.25,1,1,300.0,0.0,2,0]

model_name        = "ann"
features_to_check = "base"


path              = os.path.dirname(os.path.abspath(__file__))
model_path        = os.path.join(path, ("%s_model_%s" % (model_name,features_to_check)))
feature_file      = os.path.join(path, features_check[features_to_check]["feature_file"])
rt_countries_file = os.path.join(path, "../Datasets/ratio_tables/rt_countries_(75-25).csv")
rt_asns_file      = os.path.join(path, "../Datasets/ratio_tables/rt_asns_(75-25).csv")

y_column_idx = features_check[features_to_check]["y_column_idx"]

# Flags
recreate_model     = False # If model not created this will be concidered as True
correlation_matrix = False
point_graphs       = True

# Hyperparameters
num_malicious_samp = 1000
thresholds         = [0.5]
min_threshold	   = 0.9
poly_degree        = 3
training_epochs    = 20000
learning_rate      = 0.01
layers             = [(80,"relu"),(80,"relu"),(80,"leakyrelu"),(1,'sigmoid')]

start_time = time.time()
print("Start preprocessing")

df_features  = pd.read_csv(feature_file)
####### Append artificial data in order to fix the number of consecutive characters feature ########
if recreate_model:
	if 2 in features_check[features_to_check]["features"]:
		mal         = df_features[df_features[df_features.columns[y_column_idx]]==1].sample(500).copy()
		mal["2"]    = mal["2"].apply(lambda x:x*random.randint(3,9))
		df_features = df_features.append(mal, ignore_index=True)
####################################### END #######################################

malicious_sample      = df_features[df_features[df_features.columns[y_column_idx]]==1].sample(num_malicious_samp).copy()
malicious_sample_urls = malicious_sample['0']
malicious_sample      = malicious_sample[df_features.columns[features_check[features_to_check]["features"]]]

use_columns = features_check[features_to_check]["features"]
use_columns.append(y_column_idx)
########## Drop the manipulation malicious samples from the training phase ##########
# if recreate_model:
# 	df_features.drop(index=malicious_sample_urls.index)
######################################### END #######################################
df_features = df_features[df_features.columns[use_columns]]

elapsed_time = time.time() - start_time
print("Preprocessing time: %s" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

for threshold in thresholds:
	print("Using Threshold: %.2f" %(threshold))
	start_time = time.time()
	print("Start model building")
	if not os.path.isfile(model_path+".h5") or recreate_model:
		new_df = np.array(df_features.values)
		nn     = NeuralNetwork(dataset=new_df, learning_rate=learning_rate, threshold=threshold, training_epochs=training_epochs, degree=poly_degree)
		nn.build()
		nn.train(layers=layers)
		scores = nn.predict()
		flag   = nn.save_model(model_path)
		if not flag:
			raise ValueError("Could not save model files %s.h5 %s_scaler.sav" % (model_path))
	else:
		nn   = NeuralNetwork(threshold=threshold, degree=poly_degree)
		flag = nn.load_model(model_path)
		if not flag:
			raise ValueError("Could not find and load model files %s.h5 %s_scaler.sav" % (model_path))

	elapsed_time = time.time() - start_time
	print("Model building time: %s" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	start_time = time.time()
	print("Start manipulation")
	preds      = []
	outputs    = {}
	count      = 0
	for index, row in malicious_sample.iterrows():
		row_np = np.array([list(row)])
		pred   = nn.predict_self(row_np)[0][0][0]
		# Check if not looklike benign at first
		if pred < min_threshold:
			continue
		count += 1
		# Start features manipulation
		for change in features_check[features_to_check]["change_features"]:
			cur_check      = row_np.copy()
			feature_number = change[0]+1
			if feature_number not in outputs:
				outputs[feature_number] = {}
			for value in change[1]:
				# Changing feature: *feature_number* from value: *cur_check[0][change[0]]* to: *value*
				cur_check[0][change[0]] = value
				pred = nn.predict_self(cur_check)
				if value not in outputs[feature_number]:
					outputs[feature_number][value] = pred[0][0][0]
				else:
					outputs[feature_number][value] += pred[0][0][0]

	elapsed_time = time.time() - start_time
	print("Prediction manipulation time: %s" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	start_time = time.time()
	output = {}
	for change in features_check[features_to_check]["change_features"]:
		feature_number = change[0]+1
		output[feature_number] = []
		for value in change[1]:
			outputs[feature_number][value] /= count
			output[feature_number].append((value, outputs[feature_number][value]))

	elapsed_time = time.time() - start_time
	print("Orgainze manipulation time: %s" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	########################################## Plot point graphs ##########################################
	if point_graphs:
		start_time = time.time()
		print("Start plotting point graph")
		df  = pd.DataFrame.from_dict(output,orient ='index')
		out = []
		for i in range(0,df.shape[0]):
			l = [x for x in list(df.iloc[i]) if x is not None]
			out.append([*zip(*l)])
		cnt  = 0
		cols = int(math.sqrt(df.shape[0]))
		rows = math.ceil(float(df.shape[0])/cols)
		print("Number of samples: %d" % (count))
		print("cols: %d, rows: %d" % (cols,rows))
		fig, axs = plt.subplots(nrows=rows, ncols=cols, sharey=True, constrained_layout=True)
		for i in out:
			col = int(cnt%cols)
			row = int(cnt/cols)
			if cols>1:
				axs[row, col].plot(i[0],i[1],"-o")
				axs[row, col].set_title(features[features_check[features_to_check]["features"][cnt]])
				if col == 0:
					axs[row, col].set(ylabel='Prediction precantage')
			else:
				axs[row].plot(i[0],i[1],"-o")
				axs[row].set_title(features[features_check[features_to_check]["features"][cnt]])
				axs[row].set(ylabel='Prediction precantage')
			cnt+=1

		for col in range(cols):
			if cols>1:
				axs[rows-1, col].set(xlabel='Feature values')
			else:
				axs[rows-1].set(xlabel='Feature values')


		# Set the formatter
		formatter = FuncFormatter(to_percent)
		plt.gca().yaxis.set_major_formatter(formatter)

		plt.yticks(np.arange(0., 1.01, 0.2))
		elapsed_time = time.time() - start_time
		print("Point graphs creation time: %s" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
		plt.show()
	################################################# END #################################################

	################################### Plot correlation matrix heatmap ###################################
	if correlation_matrix:
		start_time = time.time()
		print("Start plotting correlation matrix")
		last                       = df_features.columns[-1]
		df_features.rename(columns={last:'y'},inplace=True)
		corrmat                    = df_features.corr()
		top_corr_features          = corrmat.index
		plt.figure(figsize=(12,8))
		#plot heat map
		g=sns.heatmap(df_features[top_corr_features].corr(),annot=True,cmap="RdYlGn")
		elapsed_time = time.time() - start_time
		print("Correlation matrix creation time: %s" % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
		plt.show()
	################################################# END #################################################
