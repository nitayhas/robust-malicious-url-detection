#!/usr/bin/env python

##############################################
# Author: Nitay Hason
# Extract features from dataframe
##############################################

import pandas as pd
import numpy as np
import geoip2.database
import os
import tldextract
import math
import datetime
import scipy
import time

from calendar import timegm
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from ast import literal_eval

class FeaturesExtraction:
	def __init__(self,df,countries_ratios=None,asns_ratios=None,virustotal=None,countries_threshold=-1,asns_threshold=-1,domian_idx=0,ips_idx=1,ttls_idx=2,asns_idx=3,countries_idx=4,dates_idx=6,y_idx=7,verbose=0):
		self.countries_ratios = countries_ratios
		self.countries_ratios["total_normalize"] = (self.countries_ratios["total"]-self.countries_ratios["total"].min())/(self.countries_ratios["total"].max()-self.countries_ratios["total"].min())

		self.asns_ratios = asns_ratios
		self.asns_ratios["code"] = self.asns_ratios["code"].fillna(0.0).astype(int).astype(str)
		self.asns_ratios["total_normalize"] = (self.asns_ratios["total"]-self.asns_ratios["total"].min())/(self.asns_ratios["total"].max()-self.asns_ratios["total"].min())

		self.virustotal = virustotal

		if countries_threshold < 0:
			if countries_threshold == -1:
				self.countries_threshold = int(self.countries_ratios["total"].median())
			elif countries_threshold == -2:
				self.countries_threshold = int(self.countries_ratios["total"].mean())
			elif countries_threshold == -3:
				self.countries_threshold = int(self.countries_ratios["total"].std())
			else:
				self.countries_threshold = int(scipy.stats.hmean(self.countries_ratios["total"]))
		else:
			self.countries_threshold = countries_threshold

		if asns_threshold < 0:
			if asns_threshold == -1:
				self.asns_threshold = int(self.asns_ratios["total"].median())
			elif asns_threshold == -2:
				self.asns_threshold = int(self.asns_ratios["total"].mean())
			elif asns_threshold == -3:
				self.asns_threshold = int(self.asns_ratios["total"].std())
			else:
				self.asns_threshold = int(scipy.stats.hmean(self.asns_ratios["total"]))
		else:
			self.asns_threshold = asns_threshold

		path               = os.path.dirname(os.path.abspath(__file__))
		geo_file_path = "../Datasets/GeoLite2-Country.mmdb"
		geo_file_path_full      = os.path.join(path, geo_file_path)
		self.countries_reader = geoip2.database.Reader(os.path.join(path,geo_file_path_full),['en'])
		self.df               = df
		self.verbose          = verbose
		self.merged           = {}
		self.domains_ttl      = {
			# "domain_name":{
			# 	"size": 3,
			# 	"locations": [1,8,9],
			# 	"ttls": [2388,1299,1990]
			# }
		}
		self.domian_idx    = domian_idx
		self.ips_idx       = ips_idx
		self.ttls_idx      = ttls_idx
		self.asns_idx      = asns_idx
		self.countries_idx = countries_idx
		self.dates_idx     = dates_idx
		self.y_idx         = y_idx



	def sigmoid(self,x, bias=0):
		return (1 / (1 + math.exp(-x)))+bias

	def percentage(self, percent, whole):
	  return (percent * whole) / 100.0

	### Number of consecutive characters
	def feature2(self, str):
		cur_count = 1
		count     = 0
		res       = str[0]
		n         = len(str)
		for i in range(n):
			if (i < n - 1 and
				str[i] == str[i + 1] and not str[i].isdigit()):
				cur_count += 1
			else:
				if cur_count > count:
					count = cur_count
					res   = str[i]
				cur_count = 1
		return count


	## Entropy of domain
	def feature3(self, str):
		feature3 = 0
		str_len  = len(str)
		chars    = defaultdict(int)
		for char in str:
			chars[char] += 1

		for char in str:
			pj        = (chars[char]/str_len)
			feature3 += pj*math.log(pj,2)

		return (-1)*feature3

	## Number of distinct countries
	def feature5(self, ips):
		ip_bucket = []
		countries = defaultdict(int)
		for ip in ips:
			if str(ip) not in ip_bucket:
				try:
					ip_bucket.append(str(ip))
					record = self.countries_reader.country(str(ip))
					countries[record.country.name]=1
				except:
					pass

		return len(countries)

	## Average TTL value
	def feature6(self, domain):
		avg_ttl = np.array(self.domains_ttl[domain]["ttls"]).mean()
		for l in self.domains_ttl[domain]["locations"]:
			self.merged[l][6] = avg_ttl
		return avg_ttl

	## Standard Deviation of TTL
	def feature7(self, domain):
		std_ttl = np.std(np.array(self.domains_ttl[domain]["ttls"]))
		for l in self.domains_ttl[domain]["locations"]:
			self.merged[l][7] = std_ttl
		return std_ttl

	## Countries rating
	def feature10_old(self, countries):
		rating = 0
		for country in countries:
			prec=0.75
			try:
				country_total = self.countries_ratios[self.countries_ratios["code"]==country]["total"]
				if country_total.shape[0]>0 and int(country_total.iloc[0])>=self.countries_threshold:
					prec = float(self.countries_ratios[self.countries_ratios["code"]==country]["benign_ratio"])
					# rating+=math.log(prec+0.00001,0.5)
			except Exception as exp:
				# print(country)
				# print(exp)
				pass
			rating+=math.log(prec+0.00001,0.5)
			print("Prec %.5f, Rating %.5f" % (prec, rating))
		return rating


	## Countries rating
	def feature10(self, countries):
		rating = 0
		neg = 0.00001
		for country in countries:
			prec = 0.75
			calc = 1
			cur = self.countries_ratios[self.countries_ratios["code"]==country]
			if cur.shape[0]>0:
				country_total = int(cur["total"].iloc[0])
				if country_total>=self.countries_threshold:
					calc = float(cur["total_normalize"])+neg
					prec = float(cur["benign_ratio"])
					# rating+=math.log(prec+0.00001,0.5)
			# else:
			# 	print(country)
			rating+=math.log((prec)+neg,0.5)/calc
			# print("Prec %.5f, Calc %.9f, Total %.5f, Rating %.5f" % (prec, calc, math.log((prec)+neg,0.05)/(calc+neg), rating))
		return rating


	## ASNs rating
	def feature11_old(self, asns):
		rating = 0
		for asn in asns:
			prec=0.75
			try:
				asn_total = self.asns_ratios[self.asns_ratios["code"]==asn]["total"]
				if asn_total.shape[0]>0 and int(asn_total.iloc[0])>=self.asns_threshold:
					prec = float(self.asns_ratios[self.asns_ratios["code"]==asn]["benign_ratio"])
					# rating+=math.log(prec+0.00001,0.5)
			except Exception as exp:
				# print(asn)
				# print(exp)
				pass
			rating+=math.log(prec+0.00001,0.5)
		return rating

	## ASNs rating
	def feature11(self, asns):
		rating = 0
		neg = 0.00001
		for asn in asns:
			prec = 0.75
			calc = 1
			cur = self.asns_ratios[self.asns_ratios["code"]==asn]
			if cur.shape[0]>0:
				asn_total = int(cur["total"].iloc[0])
				if asn_total>=self.asns_threshold:
					calc = float(cur["total_normalize"])+neg
					prec = float(cur["benign_ratio"])
					# rating+=math.log(prec+0.00001,0.5)
			# else:
			# 	print(asn)
			rating+=math.log((prec)+neg,0.5)/calc
		return rating

	## Expiration SSL Certificate
	def feature15(self, validity):
		not_after = self.get_epoch(validity["not_after"])
		not_before = self.get_epoch(validity["not_before"])
		res = self.exist_certificate(validity["not_after"])*(not_after-not_before)
		return res

	def get_epoch(self, time_str):
		return timegm(time.strptime(time_str, "%Y-%m-%d %H:%M:%S"))

	def exist_certificate(self, time_str):
		epoch = timegm(time.strptime(time_str, "%Y-%m-%d %H:%M:%S"))
		cur   = timegm(time.gmtime())
		return 1 if epoch>cur else 0

	def extract(self):
		x=0
		progress=0
		print("Start extraction")
		prec = self.percentage(10, len(self.df.index))
		for index, row in self.df.iterrows():
			progress+=1
			try:
				ext           = tldextract.extract(row[self.domian_idx])
				domain        = ext.domain + '.' + ext.suffix
				ips_arr       = literal_eval(row[self.ips_idx])
				ips_arr		  = list(set(ips_arr))
				ttls_arr      = literal_eval(row[self.ttls_idx])
				asns_arr      = literal_eval(row[self.asns_idx])
				countries_arr = literal_eval(row[self.countries_idx])
				dates_arr     = literal_eval(row[self.dates_idx])
				if domain in self.domains_ttl:
					self.domains_ttl[domain]["size"] +=1
				else:
					self.domains_ttl[domain] = {
						"size": 1,
						"locations": [],
						"ttls": []
					}
				for ttl in ttls_arr:
					self.domains_ttl[domain]["ttls"].append(ttl)

				benign_malicious = row[self.y_idx]
				creation_date    = int(dates_arr[0])
				expiration_date  = int(dates_arr[1])
				updated_date     = int(dates_arr[2])
				feature1_str     = len(domain)
				feature2_str     = self.feature2(domain)
				feature3_str     = self.feature3(domain)
				feature4_str     = len(ips_arr)
				feature5_str     = self.feature5(ips_arr)
				feature6_str     = self.feature6(domain)
				feature7_str     = self.feature7(domain)
				feature8_str     = relativedelta(datetime.datetime.fromtimestamp(expiration_date), datetime.datetime.fromtimestamp(creation_date)).years
				feature9_str     = relativedelta(datetime.datetime.fromtimestamp(updated_date), datetime.datetime.fromtimestamp(creation_date)).years
				feature10_str    = self.feature10(countries_arr)
				feature11_str    = self.feature11(asns_arr)
				if self.virustotal is not None:
					if row[self.domian_idx] in self.virustotal.index:
						feature12_str = len(self.virustotal.loc[row[self.domian_idx]]["dns_records"]) if self.virustotal.loc[row[self.domian_idx]]["dns_records"] is not np.nan else 0
						feature13_str = len(self.virustotal.loc[row[self.domian_idx]]["resolutions"]) if self.virustotal.loc[row[self.domian_idx]]["resolutions"] is not np.nan else 0
						feature14_str = len(self.virustotal.loc[row[self.domian_idx]]["subdomains"]) if self.virustotal.loc[row[self.domian_idx]]["subdomains"] is not np.nan else 0
						try:
							feature15_str = self.feature15(self.virustotal.loc[row[self.domian_idx]]["last_https_certificate"]["validity"])
						except Exception as exp:
							feature15_str = 0
						try:
							feature16_str = self.exist_certificate(self.virustotal.loc[row[self.domian_idx]]["last_https_certificate"]["validity"]["not_after"])
						except:
							feature16_str = 0
					else:
						feature12_str = 0
						feature13_str = 0
						feature14_str = 0
						feature15_str = 0
						feature16_str = 0
					self.merged[x]   = {0: str(row[self.domian_idx]), 1:feature1_str, 2:feature2_str, 3:feature3_str, 4:feature4_str, 5:feature5_str, 6:feature6_str, 7:feature7_str, 8:feature8_str, 9:feature9_str, 10:feature10_str, 11:feature11_str, 12:feature12_str, 13: feature13_str, 14: feature14_str, 15: feature15_str, 16:feature16_str, 17: str(benign_malicious)}
				else:
					self.merged[x]   = {0: str(row[self.domian_idx]), 1:feature1_str, 2:feature2_str, 3:feature3_str, 4:feature4_str, 5:feature5_str, 6:feature6_str, 7:feature7_str, 8:feature8_str, 9:feature9_str, 10:feature10_str, 11:feature11_str, 12:str(benign_malicious)}
				self.domains_ttl[domain]["locations"].append(x)
				x+=1
			except Exception as exp:
				if self.verbose > 1:
					print(exp)
				pass
			# print(merged)
			if self.verbose > 0:
				if self.verbose > 1:
					print(progress, end=',',flush=True)
					if int(progress%prec) == 0:
						print(int(progress/prec)*10,flush=True)
				elif int(progress%prec) == 0:
					print(int(progress/prec)*10, end=',',flush=True)
		self.df_extracted = pd.DataFrame.from_dict(self.merged, "index")
		print("\nFinish extraction")

	def to_csv(self,path):
		if self.df_extracted is not None:
			self.df_extracted.to_csv(path, encoding='utf-8', index=False)
		else:
			print("Feature was not extracted yet")
