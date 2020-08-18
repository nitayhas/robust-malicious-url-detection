#!/usr/bin/env python

##############################################
# Author: Nitay Hason
# Full flow with multiple ratio tables thresholds: preprocessing, feature extraction and model training.
##############################################

from Tools import *
import numpy as np
import pandas as pd
import time
import os
from fractions import Fraction

from Tools.RatioMatrix import RatioMatrix
from Tools.FeaturesExtraction import FeaturesExtraction
from Models.Supervised.NeuralNetwork import NeuralNetwork
from sklearn.utils import shuffle

path           = os.path.dirname(os.path.abspath(__file__))

url_column_idx = 0
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


### precantage for ratio tables and features extraction dataset
ratio_prec    = 0.
features_prec = 1-ratio_prec

######################################### Thresholds list to check #########################################
# Dictionary structure:
# "Threshold Name":(countries_threshold, asns_threshold)
#
# Threshold Types: Median: -1, Mean (Avg): -2, Standard Deviation: -3, Harmonic Mean: -4, Custom: i>=0
# thresholds_lst = {"median":(-1,-1),"mean":(-2,-2),"std":(-3,-3),"hmean":(-4,-4),"nothresholds":(0,0)}
# thresholds_lst = {"median":(9,2),"mean":(306,33),"std":(1855,480),"hmean":(3,1),"nothresholds":(0,0)}
thresholds_lst = {"median":(9,2)}
# Example of custom thresholds: thresholds_lst = {"median":(9,2),"mean":(306,33),"std":(1855,480),"hmean":(3,1),"nothresholds":(0,0)}
##############################################################################################################

features_set     = "hybrid_robust"
ann_threshold    = 0.5
learning_rate    = 0.001
training_epochs  = 20000
degree           = 1
test_size		 = 0.25


outputs = {}
for thresholds_type, thresholds_values in thresholds_lst.items():
	countries_threshold = thresholds_values[0]
	asns_threshold      = thresholds_values[1]
	print("Threshold type - %s, Countries threshold - %d, ASNs threshold - %d" % (thresholds_type, countries_threshold, asns_threshold))

	ben_prec = 0.75
	mal_prec = 1-ben_prec # 0.25

	preprocessing_stage           = True
	ratio_tables_stage            = True
	features_extraction_stage     = True
	virustotal_features_exist	  = True
	training_model_stage          = False

	if not features_extraction_stage and (countries_threshold<0 or asns_threshold<0):
		raise ValueError("You must give thresholds larger than 0")

	dns_for_ratios_file           = "temp_dns_for_ratios_%d_perc.csv" % (int(ratio_prec*100))
	dns_for_features_file         = "temp_dns_for_features_%d_perc.csv" % (int(features_prec*100))
	benign_ds                     = "../Datasets/dns/dns_benign_top_5345_urlscan.csv"
	malicious_ds                  = "../Datasets/dns/dns_malicious_phishtank_urlscan.csv"
	malicious_ds_old              = "../Datasets/dns/dns_malicious_various_urlscan.csv"
	virustotal_ds                 = "../Datasets/virustotal/vt_benign_5345_malicious_1356.json"
	rm_countries_file             = "../Datasets/ratio_tables/countries_(%d-%d).csv" % (int(ratio_prec*100),int(features_prec*100))
	rm_asns_file                  = "../Datasets/ratio_tables/asns_(%d-%d).csv" % (int(ratio_prec*100),int(features_prec*100))
	temp_features_extraction_file = "../Datasets/features_extractions/%s_%d_%d_(%d-%d)%s.csv" % (thresholds_type,countries_threshold,asns_threshold,int(ratio_prec*100),int(features_prec*100), "_vt_include" if virustotal_features_exist else "")

	if preprocessing_stage:
		#################################################
		################ PREPROCESSING ##################
		#################################################
		df_ben     = pd.read_csv(os.path.join(path, benign_ds), sep        = ';', header =None, dtype =str)
		df_mal     = pd.read_csv(os.path.join(path, malicious_ds), sep     = ';', header =None, dtype =str)
		df_mal_old = pd.read_csv(os.path.join(path, malicious_ds_old), sep = ';', header =None, dtype =str)

		domains_list_ben     = pd.DataFrame(df_ben[0].unique())
		domains_list_mal_new = pd.DataFrame(df_mal[0].unique())
		domains_list_mal_old = pd.DataFrame(df_mal_old[0].unique())
		domains_list_mal     = pd.concat([domains_list_mal_new,domains_list_mal_old])

		domains_list_ben = shuffle(domains_list_ben)
		domains_list_mal = shuffle(domains_list_mal)

		print("Total Ben - %d" % (domains_list_ben.shape[0]))
		print("Total Mal - %d" % (domains_list_mal.shape[0]))
		print("New Mal   - %d" % (domains_list_mal_new.shape[0]))
		print("Old Mal   - %d" % (domains_list_mal_old.shape[0]))

		frac_mal = Fraction(mal_prec).denominator
		##### list1 for ratio tables, list2 for feature extraction
		ratio_mal = domains_list_mal.shape[0]*ratio_prec
		domains_list_mal1    = domains_list_mal.sample(int(ratio_mal))
		domains_list_ben1    = domains_list_ben.sample(int(ratio_mal*frac_mal*ben_prec))
		domains_list_mal2  	 = domains_list_mal[~domains_list_mal[0].isin(domains_list_mal1[0])]
		domains_list_ben2  	 = domains_list_ben[~domains_list_ben[0].isin(domains_list_ben1[0])]

		ben_samples = int(domains_list_mal2.shape[0]*frac_mal*ben_prec)
		if ben_samples<domains_list_ben2.shape[0]:
			domains_list_ben2  	 = domains_list_ben2.sample(ben_samples)

		df_mal_ratios = pd.concat([df_mal[df_mal[0].isin(domains_list_mal1[0])],df_mal_old[df_mal_old[0].isin(domains_list_mal1[0])]])
		df_ben_ratios = df_ben[df_ben[0].isin(domains_list_ben1[0])]

		df_mal_features = pd.concat([df_mal[df_mal[0].isin(domains_list_mal2[0])],df_mal_old[df_mal_old[0].isin(domains_list_mal2[0])]])
		df_ben_features = df_ben[df_ben[0].isin(domains_list_ben2[0])]

		df_ratios   = pd.concat([df_mal_ratios,df_ben_ratios])
		df_ratios   = df_ratios.drop_duplicates(subset=0, keep="last")
		df_features = pd.concat([df_mal_features,df_ben_features])
		df_features = df_features.drop_duplicates(keep="last")

		print("Ratio Ben - %d , Ratio Mal - %d" % (df_ben_ratios[0].unique().shape[0],df_mal_ratios[0].unique().shape[0]))
		print("Features Ben - %d , Features Mal - %d" % (df_ben_features[0].unique().shape[0],df_mal_features[0].unique().shape[0]))
		print("Total Ratio - %d" % (df_ratios.shape[0]))
		print("Total Features - %d" % (df_features.shape[0]))

		# df_ratios.to_csv(os.path.join(path, dns_for_ratios_file), sep=';', encoding='utf-8', index=False)
		# df_features.to_csv(os.path.join(path, dns_for_features_file), sep=';', encoding='utf-8', index=False)
		#################################################
		##################### END #######################
		#################################################
	elif ratio_tables_stage:
		df_ratios   = pd.read_csv(os.path.join(path, dns_for_ratios_file), sep   =';')
		df_features = pd.read_csv(os.path.join(path, dns_for_features_file), sep =';')

	if ratio_tables_stage:
		#################################################
		############ Ratio tables creation ##############
		#################################################
		rm_countries = RatioMatrix(df_ratios, 4)
		rm_asns      = RatioMatrix(df_ratios, 3)

		rm_countries.extract()
		rm_asns.extract()
		# rm_countries.to_csv(os.path.join(path, rm_countries_file))
		# rm_asns.to_csv(os.path.join(path, rm_asns_file))

		countries_ratios = rm_countries.df_extracted
		asns_ratios      = rm_asns.df_extracted
		#################################################
		##################### END #######################
		#################################################
	elif features_extraction_stage:
		countries_ratios = pd.read_csv(os.path.join(path, rm_countries_file), sep     =';')
		asns_ratios      = pd.read_csv(os.path.join(path, rm_asns_file), sep          =';')
		df_features      = pd.read_csv(os.path.join(path, dns_for_features_file), sep =';')

	if features_extraction_stage:
		#################################################
		############### Feature extraction ###############
		#################################################
		if virustotal_features_exist:
			df_virustotal 	                        = pd.read_json(os.path.join(path, virustotal_ds))
			df_virustotal                           = df_virustotal.T
			na_cert                                 = {"validity":{"not_before":"1970-01-01 12:00:00","not_after":"1970-01-01 12:00:00"},"issuer":{"O":"NA"}}
			df_virustotal['last_https_certificate'] = df_virustotal['last_https_certificate'].apply(lambda x: na_cert if x != x else x)
			vt_use_cols                             = ['dns_records','last_https_certificate','resolutions','subdomains']
			df_virustotal                           = df_virustotal[vt_use_cols]
		else:
			df_virustotal = None

		features_extraction = FeaturesExtraction(df_features, countries_ratios=countries_ratios, asns_ratios=asns_ratios, virustotal=df_virustotal, countries_threshold=countries_threshold, asns_threshold=asns_threshold,domian_idx=0,ips_idx=1,ttls_idx=2,asns_idx=3,countries_idx=4,dates_idx=9,y_idx=10, verbose=1)
		print('Countries threshold: %d' % features_extraction.countries_threshold)
		print('ASNs threshold: %d' % features_extraction.asns_threshold)
		features_extraction.extract()
		features_extraction.to_csv(os.path.join(path,("../Datasets/features_extractions/%s_%d_%d_(%d-%d)%s.csv" % (thresholds_type,features_extraction.countries_threshold,features_extraction.asns_threshold,int(ratio_prec*100),int(features_prec*100), "_vt_include" if virustotal_features_exist else ""))))
		df_features = features_extraction.df_extracted.copy()
		print("Benign")
		print(df_features[df_features[df_features.columns[y_column_idx]]==0].describe())
		print("Malicious")
		print(df_features[df_features[df_features.columns[y_column_idx]]==1].describe())
		#################################################
		##################### END #######################
		#################################################
	elif training_model_stage:
		df_features = pd.read_csv(os.path.join(path, temp_features_extraction_file))

	if training_model_stage:
		#################################################
		################# Model Training ################
		#################################################
		df_features = df_features[(df_features[df_features.columns[8]]!=0) & (df_features[df_features.columns[9]]!=0)]
		df_features[df_features.columns[y_column_idx]] = df_features[df_features.columns[y_column_idx]].astype(int)
		domains_list_mal = df_features[df_features[df_features.columns[y_column_idx]]==1].shape[0]

		df_features      = df_features.dropna()
		df_features_new  = df_features[df_features[df_features.columns[y_column_idx]]==0].copy()
		df_features_new  = df_features_new.drop_duplicates(subset=df_features_new.columns[url_column_idx], keep ="last")
		df_features_conc = df_features[df_features[df_features.columns[y_column_idx]]==0].sample(3*domains_list_mal - df_features_new.shape[0]).copy()# if 3*domains_list_mal<= df_features[df_features[df_features.columns[-2]]!=0].shape[0] else df_features[~df_features.isin(df_features_new)].copy()
		df_features_new  = pd.concat([df_features_new,df_features_conc,df_features[df_features[df_features.columns[y_column_idx]]==1].copy()])
		df_features_new  = df_features_new.dropna()

		use_columns = features_check[features_set]["features"]
		use_columns.append(y_column_idx)
		new_df      = df_features_new[df_features_new.columns[use_columns]]
		new_df      = np.array(new_df.values)

		nn          = NeuralNetwork(dataset=new_df, threshold=ann_threshold, learning_rate=learning_rate, training_epochs=training_epochs, test_size=test_size)
		nn.build()
		nn.train(layers=[(80,"relu"),(80,"relu"),(80,"leakyrelu"),(1,'sigmoid')])
		scores = nn.predict()
		del scores["classification_report"]
		scores["confusion_matrix"] = str(scores["confusion_matrix"]).strip()
		outputs[thresholds_type] = scores

if training_model_stage:
	out = pd.DataFrame.from_dict(outputs,orient='index')
	out.to_pickle(os.path.join(path, ("temp_pickle_(%d-%d).pkl" % (int(ratio_prec*100),int(features_prec*100)))))
	out.to_csv(os.path.join(path, ("temp_output_(%d-%d).csv" % (int(ratio_prec*100),int(features_prec*100)))), encoding="utf-8", sep=";")
