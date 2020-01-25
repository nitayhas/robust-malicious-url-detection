#!/usr/bin/env python

##############################################
# Author: Nitay Hason
# Packet sniffer in order to fetch the appropriate data.
##############################################

from scapy.all import *
from threading import Thread, Event, Lock

class Sniffer(Thread):
	def  __init__(self, filter=""):
		super().__init__()
		self.filter          = filter
		self.stop_sniffer    = Event()
		self.mutex           = Lock()
		self.last_packet     = None
		self.last_packet_id  = 0
		self.check_url    = ""
		self.urls_list    = []
		self.urls_packets = {}

	def run(self):
		sniff(filter=self.filter, prn=self.print_packet, stop_filter=self.should_stop_sniffer)

	def should_stop_sniffer(self, packet):
		return self.stop_sniffer.isSet()

	def join(self, timeout=None):
		self.stop_sniffer.set()
		super().join(timeout)

	def print_packet(self, packet):
		dns_layer = packet.getlayer(DNS)
		if DNSRR in packet:
			self.mutex.acquire()
			lst = self.urls_list
			self.mutex.release()
			qname = str(dns_layer.qd.qname.decode("utf-8"))[:-1]
			try:
				ind = next(i for i,v in enumerate(lst) if qname in v)
			except:
				ind = -1
			if ind>-1:
				self.append_packet(lst[ind], packet)
				# self.last_packet_id+=1
				# self.last_packet = packet
			# if self.check_url in str(dns_layer.qd.qname):
			# 	self.last_packet_id+=1
			# 	self.last_packet = packet

	def urls_count(self):
		self.mutex.acquire()
		count = len(self.urls_list)
		self.mutex.release()
		return count

	def packet_id(self):
		self.mutex.acquire()
		id = self.last_packet_id
		self.mutex.release()
		return id

	def append_url(self, url):
		self.mutex.acquire()
		self.urls_list.append(url)
		self.mutex.release()

	def remove_url(self, url):
		self.mutex.acquire()
		try:
			self.urls_list.remove(url)
		except:
			pass
		self.mutex.release()

	def append_packet(self, url, packet):
		self.mutex.acquire()
		self.urls_packets[url] = packet
		self.last_packet_id=len(self.urls_packets)
		self.mutex.release()

	def pop_packet(self, url=None):
		if self.packet_id()>0:
			self.mutex.acquire()
			if url is None:
				url = list(self.urls_packets.keys())[-1]
			packet = self.urls_packets[url]
			try:
				del self.urls_packets[url]
			except:
				pass
			try:
				self.urls_list.remove(url)
			except:
				pass
			self.last_packet_id=len(self.urls_packets)
			self.mutex.release()
		else:
			packet=None
		return (url,packet)
