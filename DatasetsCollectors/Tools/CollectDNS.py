#!/usr/bin/env python

##############################################
# Author: Nitay Hason
# DNS collector class
##############################################

import schedule
import time
import random
import json
import datetime
import requests
import threading
import logging
import os
import pandas as pd
import numpy as np
from Sniffer import Sniffer
from scapy.all import DNS, DNSRR
from time import sleep


class CollectDNS:
	def __init__(self, df, dns_timeout=4.0, output='output.csv', domains_tmp='domains_tmp.json', tmp_file="df_tmp.pkl", log_file="log_file.log"):
		# self.df               = df
		self.dns_timeout      = dns_timeout
		self.output           = output
		self.tmp_file         = tmp_file
		self.domains_tmp      = domains_tmp
		self.merged           = {}
		self.domains_to_check = {}
		self.sniffer_flag     = False
		self.check_sniffing   = True
		self.appended         = 0

		for index, row in df.iterrows():
			if row[0] not in self.domains_to_check:
				self.domains_to_check[row[0]] = {"status": row[(df.shape[1]-1)], "fails":0}


		logging.basicConfig(filename=log_file, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)

	def percentage(self, percent, whole):
	  return (percent * whole) / 100.0

	def saveData(self):
		if os.path.isfile(self.output):
			# with open(output, 'a') as f:
			df = pd.DataFrame.from_dict(self.merged, "index")
			df.to_csv(self.output, mode='a',header=None, encoding='utf-8', sep = ';', index=False)
		else:
			df = pd.DataFrame.from_dict(self.merged, "index")
			df.to_csv(self.output, header=None, encoding='utf-8', sep = ';', index=False)

		self.merged.clear()

	def saveTmp(self):
		df = pd.DataFrame.from_dict(self.merged, "index")
		df.to_pickle(self.tmp_file)

	def removeTmp(self):
		if os.path.isfile(self.tmp_file):
		    os.remove(self.tmp_file)

	def sendHead(self, domain):
		url = domain
		try:
			if url.find("http://") == url.find("https://"):
				url = "http://"+domain
				try:
					req = requests.head(url,timeout=3)
				except:
					url = "https://"+domain
					req = requests.head(url,timeout=3)
			else:
				req = requests.head(url,timeout=3)
			ret = req.ok
		except Exception as exp:
			logging.error("try sniffed domain " + domain + " got exception: " + str(exp))
			ret = False
		return ret

	def checkSniffing(self, sniffer):
		print("Checking Sniffing", flush=True)
		logging.info("Checking Sniffing")
		self.check_sniffing = True
		while(sniffer.urls_count()>0 and self.check_sniffing):
			if sniffer.packet_id()>0:
				packet = sniffer.pop_packet()
				domain = packet[0]
				print("Found %s" % domain, flush=True)
				logging.info(domain)
				dns_resp = packet[1].getlayer(DNS)
				ips = []
				ttls = []
				for x in range(dns_resp[DNS].ancount):
					ttls.append(dns_resp[DNSRR][x].ttl)
					ips.append(dns_resp[DNSRR][x].rdata)

				if len(ips)>0 and len(ttls)>0:
					self.merged[len(self.merged)] = {0:domain,1:ips,2:ttls,3:self.domains_to_check[domain]["status"]}
					self.saveTmp()
					self.appended+=1
				self.domains_to_check[domain]["fails"] = 0
			sleep(0.5)
		print("Finish checking Sniffing", flush=True)

	def startSniff(self):
		if self.sniffer_flag is False:
			self.sniffer_flag = True
			cur_id = 0
			sniffer = Sniffer(filter="udp port 53")
			check_sniffer = threading.Thread(target=self.checkSniffing, args=(sniffer,))
			check_sniffer.daemon = True

			domains = list(self.domains_to_check.keys())
			random.shuffle(domains)

			print("[*] Start sniffing...")
			logging.info("Start sniffing...")
			sniffer.start()
			progress=0
			self.appended=0
			prec = self.percentage(10, len(domains))
			for domain in domains:
				progress+=1
				# print(domain, flush=True)
				try:
					exp_flag = False
					sniffer.append_url(domain)
					if sniffer.urls_count()>1:
						if not check_sniffer.is_alive():
							self.check_sniffing = False
							sleep(2)
							check_sniffer = threading.Thread(target=self.checkSniffing, args=(sniffer,))
							check_sniffer.daemon = True
							check_sniffer.start()
					# sniffer.check_domain = domain
					exp_flag = self.sendHead(domain)
					sleep(2)
					if not exp_flag:
						sniffer.remove_url(domain)
						self.domains_to_check[domain]["fails"] = self.domains_to_check[domain]["fails"]+1
						if self.domains_to_check[domain]["fails"] > 3:
							del self.domains_to_check[domain]

				except KeyboardInterrupt:
					print("[*] Stop sniffing 2", flush=True)
					logging.info("Stop sniffing by KeyboardInterrupt, total: " + str(len(self.domains_to_check.keys())) + " appended: " + str(self.appended))
					sniffer.join(1)
					break
				except Exception as exp:
					print(exp, flush=True)
					logging.error("try sniffed domain " + domain + " got exception: " + str(exp))
				if int(progress%prec) == 0:
					self.saveData()
					self.removeTmp()
					print(int(progress/prec)*10, end=',', flush=True)
			print("Finish domains to check loop", flush=True)
			self.saveData()
			self.removeTmp()
			check_sniffer.join(1)
			print("[*] Stop sniffing 1", flush=True)
			logging.info("Stop sniffing, total: " + str(len(self.domains_to_check.keys())) + " appended: " + str(self.appended))
			sniffer.join(1)
			self.sniffer_flag = False
