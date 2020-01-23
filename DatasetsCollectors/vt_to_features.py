#!/usr/bin/env python

##############################################
# Author: Nitay Hason
# Extract from VirusTotal json file the appropriate data
##############################################

import pandas as pd
import numpy as np
import time
from calendar import timegm

df_ben 	 = pd.read_json("../Datasets/virustotal/vt_benign_5345.json")
df_mal 	 = pd.read_json("../Datasets/virustotal/vt_malicious_1356.json")

df_ben   = df_ben.T
df_mal   = df_mal.T

na_cert = {"validity":{"not_after":"1970-01-01 12:00:00"},"issuer":{"O":"NA"}}
df_ben['last_https_certificate'] = df_ben['last_https_certificate'].apply(lambda x: na_cert if x != x else x)
df_mal['last_https_certificate'] = df_mal['last_https_certificate'].apply(lambda x: na_cert if x != x else x)

use_cols = ['dns_records','dns_records_date','https_certificate_date','last_https_certificate','resolutions','subdomains']
new_cols = ['resolutions_count', 'subdomains_count', 'dns_records_count', 'https_validity', 'ssl_exists']

df_ben["resolutions_count"] = df_ben["resolutions"].str.len()
df_mal["resolutions_count"] = df_mal["resolutions"].str.len()
df_ben["subdomains_count"]  = df_ben["subdomains"].str.len()
df_mal["subdomains_count"]  = df_mal["subdomains"].str.len()
df_ben["dns_records_count"] = df_ben["dns_records"].str.len()
df_mal["dns_records_count"] = df_mal["dns_records"].str.len()

df_ben["resolutions_count"].fillna(0,inplace=True)
df_mal["resolutions_count"].fillna(0,inplace=True)
df_ben["subdomains_count"].fillna(0,inplace=True)
df_mal["subdomains_count"].fillna(0,inplace=True)
df_ben["dns_records_count"].fillna(0,inplace=True)
df_mal["dns_records_count"].fillna(0,inplace=True)


def get_epoch(validity):
    epoch = 0
    if "not_after" in validity:
        epoch = timegm(time.strptime(validity["not_after"], "%Y-%m-%d %H:%M:%S"))
    return epoch

def exist_certificate(validity):
    epoch = 0
    cur   = 0
    if "not_after" in validity:
        epoch = timegm(time.strptime(validity["not_after"], "%Y-%m-%d %H:%M:%S"))
        cur   = timegm(time.gmtime())
    return 1 if epoch>cur else 0


df_ben["https_validity"] = df_ben["last_https_certificate"].str["validity"].apply(get_epoch)
df_mal["https_validity"] = df_mal["last_https_certificate"].str["validity"].apply(get_epoch)
df_ben["ssl_exists"] = df_ben["last_https_certificate"].str["validity"].apply(exist_certificate)
df_mal["ssl_exists"] = df_mal["last_https_certificate"].str["validity"].apply(exist_certificate)


df_ben[new_cols].to_csv("../Datasets/virustotal/vt_benign_5345_features.csv")
df_mal[new_cols].to_csv("../Datasets/virustotal/vt_malicious_1356_features.csv")
