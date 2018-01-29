import tensorflow as tf
import numpy as np
from agent1 import Worker
import threading 
from brain import Master
from network import NetworkGenerator
import time
class control:

	def __init__(self):
		self.num_workers=1
		

	def spawnWorkers(self):
		
		for i in range(self.num_workers):
			workerObj1=Worker(i+1)

			workerObj1.start()
			time.sleep(1)

mastObj=Master(0)#Initially global network is created and it is saved to a location------#
conObj=control()#The worker threads are created,each copying global network----------#

mastObj.start()
time.sleep(2)
conObj.spawnWorkers()

