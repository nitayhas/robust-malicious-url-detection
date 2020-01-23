#!/usr/bin/env python

##############################################
# Author: Nitay Hason
# Create ratio matrix
##############################################

import pandas as pd
from ast import literal_eval

class RatioMatrix:
	def __init__(self, df, row_number):
		self.df           = df
		self.row_number   = row_number
		self.df_extracted = None

	def extract(self):
		dictionary = {}
		last_index = self.df.shape[1]-1
		for index,row in self.df.iterrows():
			arr = literal_eval(row[self.row_number])
			for item in arr:
				if item not in dictionary:
					if row[last_index] == '1':
						dictionary[item] = {"benign": 0, "malicious": 1}
					else:
						dictionary[item] = {"benign": 1, "malicious": 0}
				else:
					if row[last_index] == '1':
						dictionary[item]["malicious"] += 1
					else:
						dictionary[item]["benign"]    += 1

		self.df_extracted                    = pd.DataFrame.from_dict(dictionary, orient ='index')
		self.df_extracted["benign_ratio"]    = self.df_extracted["benign"] / (self.df_extracted["benign"]+self.df_extracted["malicious"])
		self.df_extracted["malicious_ratio"] = self.df_extracted["malicious"] / (self.df_extracted["benign"]+self.df_extracted["malicious"])
		self.df_extracted["total"]           = self.df_extracted["benign"] + self.df_extracted["malicious"]
		self.df_extracted["code"]            = self.df_extracted.index
		self.df_extracted                    = self.df_extracted[["code", "benign_ratio", "malicious_ratio","benign","malicious","total"]]

	def to_csv(self, path):
		if self.df_extracted is not None:
			try:
				self.df_extracted.to_csv(path, encoding='utf-8', sep=';', index=False)
			except Exception as exp:
				print(exp)
		else:
			print("Feature was not extracted yet")
