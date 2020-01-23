#!/usr/bin/env python

##############################################
# Author: Nitay Hason
# Append urlscan.io information to the dns dataset
##############################################

from DatasetsCollectors.Tools.UrlScan import *
import pandas as pd
import time
import json
import whois
import tldextract
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a','--apikey', help='API key for Urlscan')
parser.add_argument('-i','--csv_in', help='CSV input file datatset')
parser.add_argument('-o','--csv_out', help='CSV output file datatset')
parser.add_argument('-d','--dataframe', help='Dataframe save file')

args = parser.parse_args()
path = os.path.dirname(os.path.abspath(__file__))

def checkWhois(domain):
	try:
		whos = whois.whois(domain)
		if whos.creation_date is not None and whos.expiration_date is not None and whos.updated_date is not None:
			creation_date = whos.creation_date if not isinstance(whos.creation_date, (list,)) else whos.creation_date[0]
			expiration_date = whos.expiration_date if not isinstance(whos.expiration_date, (list,)) else whos.expiration_date[0]
			updated_date = whos.updated_date if not isinstance(whos.updated_date, (list,)) else whos.updated_date[0]
			return [creation_date.timestamp(),expiration_date.timestamp(),updated_date.timestamp()]
	except Exception as exp:
		print(exp)
	return []

def fixCountries(asns):
	countries      = []
	asns_countries = {}
	with open(os.path.join(path, '../Datasets/asns/asns_countries.json'), 'r') as f:
		asns_countries = json.load(f)
	for asn in asns:
		try:
			country = asns_countries[asn]
			countries.append(country)
		except:
			pass
	return countries

API_KEY        = args.apikey
csv_in         = args.csv_in
csv_out        = args.csv_out
dataframe_save = "df_save" if args.dataframe is None else args.dataframe

domains        = []
domains_data   = {}


if os.path.isfile(dataframe_save):
	df       = pd.read_pickle(dataframe_save)
	ind      = len(df.columns)-8
	last_ind = df[df[1]!='[]'].iloc[-1].name
	domains  = list(df[0].iloc[:(last_ind+1)])

	for index, row in df.iloc[:(last_ind+1)].iterrows():
		asns                 = df.at[index,ind]
		countries            = df.at[index,ind+1]
		ips                  = df.at[index,ind+2]
		domains_scans        = df.at[index,ind+3]
		urls                 = df.at[index,ind+4]
		servers              = df.at[index,ind+5]
		whos                 = df.at[index,ind+6]
		domains_data[row[0]] = [asns,countries,ips,domains_scans,urls,servers,whos]
else:
	df             = pd.read_csv(csv_in, sep =";", header=None, dtype=str)
	ind            = len(df.columns)-1
	df             = df.rename(columns={ind:ind+7})
	df[ind]        = '[]'
	df[ind+1]      = '[]'
	df[ind+2]      = '[]'
	df[ind+3]      = '[]'
	df[ind+4]      = '[]'
	df[ind+5]      = '[]'
	df[ind+6]      = '[]'
	df             = df.reindex(sorted(df.columns), axis=1)

for index, row in df.iterrows():
	print(index, end=",", flush=True)
	try:
		if row[0] not in domains:
			ext = tldextract.extract(row[0])
			domain = ext.domain + '.' + ext.suffix
			whos = []
			u = UrlScan(apikey=API_KEY,url=row[0])
			#Starting a scan
			try:
				u.submit() #Wait a few seconds for the scan to complete, you can check with u.checkStatus()
				result = None
				while result is None:
					try:
						result = u.getJson()
						if 'message' in result:
							if result['message'] == 'notdone':
								result = None
								raise Exception('Not done')
					except Exception as exp:
						time.sleep(1)
						pass

				if result is not None:
					try:
						result["lists"]["countries"] = fixCountries(result["lists"]["asns"])
					except:
						pass
				if ext.suffix != '':
					whos = checkWhois(domain)
				else:
					print(ext)
				asns               = result["lists"]["asns"]
				countries          = result["lists"]["countries"]
				ips                = result["lists"]["ips"]
				domains_scans      = result["lists"]["domains"]
				urls               = result["lists"]["urls"]
				servers            = result["lists"]["servers"]
				df.at[index,ind]   = str(asns)
				df.at[index,ind+1] = str(countries)
				df.at[index,ind+2] = str(ips)
				df.at[index,ind+3] = str(domains_scans)
				df.at[index,ind+4] = str(urls)
				df.at[index,ind+5] = str(servers)
				df.at[index,ind+6] = str(whos)

				domains.append(row[0])
				domains_data[row[0]] = [asns,countries,ips,domains_scans,urls,servers,whos]

			except Exception as exp:
				print(exp)
		else:
			df.at[index,ind]   = str(domains_data[row[0]][0])
			df.at[index,ind+1] = str(domains_data[row[0]][1])
			df.at[index,ind+2] = str(domains_data[row[0]][2])
			df.at[index,ind+3] = str(domains_data[row[0]][3])
			df.at[index,ind+4] = str(domains_data[row[0]][4])
			df.at[index,ind+5] = str(domains_data[row[0]][5])
			df.at[index,ind+6] = str(domains_data[row[0]][6])

		df.to_pickle(dataframe_save)
	except Exception as exp:
		print(exp)


df.to_csv(csv_out, encoding='utf-8', sep = ';', index=False, header=None)
## If dataframe_save exists, delete it
if os.path.isfile(dataframe_save):
    os.remove(dataframe_save)
else:
    print("Error: %s file not found" % dataframe_save)
print("FINISH")
