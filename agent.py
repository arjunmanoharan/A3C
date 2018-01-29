import tensorflow as tf
import numpy as np
import threading
import time
from brain import Master
from collections import deque
import math
from keras.models import *
from keras.layers import *
from keras import backend as K
import h5py
class Worker(threading.Thread):
	MIN_EPSILON=0.1
	MAX_EPSILON=1.0
	LAMBDA=0.01
	def __init__(self,thread_no):
		threading.Thread.__init__(self)
		self.lock=threading.Lock()
		self.gamma=0.99
		self.n_step_return=3
		self.num_epsiodes=2
		self.graph=tf.Graph()
		self.epsilon=1.0
		self.action_num=2			
		self.worker_env=gym.make('Cartpole-v0')
		self.agent_no=thread_no
		self.mastObj=Master(thread_no)
		self.memory=deque(maxlen=100)
		self.trainingQ=deque(maxlen=100)
		
		
		self.opt,self.loss_total,self.inputs,self.a_t,self.r_t,self.sess,self.model,self.value=self.buildModel1()
		self.loadingModel()
		
		print("##################worker n/w$#########################")


	def buildModel1(self):
		
		a_t=Input(batch_shape=(None,2))		
		r_t=Input(batch_shape=(None,1))
		sess1=tf.Session()

		l_input = Input( batch_shape=(None, 4) )
		l_dense = Dense(16, activation='relu')(l_input)

		policy = Dense(2, activation='softmax')(l_dense)
		value   = Dense(1, activation='linear')(l_dense)

		model = Model(inputs=[l_input], outputs=[policy, value])
		model._make_predict_function()


		log_prob = tf.log( tf.reduce_sum(tf.multiply(policy,a_t), axis=1, keep_dims=True))
		advantage = tf.subtract(r_t ,value)

		loss_policy = - tf.multiply(log_prob,advantage)									
		loss_value  = 0.5 * tf.square(advantage)												
		entropy = 0.01 * tf.reduce_sum(tf.multiply(policy , tf.log(policy)), axis=1, keep_dims=True)	

		loss_total = tf.reduce_mean(entropy)
		
		opt=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_total)	
		init=tf.global_variables_initializer()
		init1=tf.local_variables_initializer()
		model._make_predict_function()
		
		
		sess1.run(init)
		sess1.run(init1)
		
		return opt,loss_total,l_input,a_t,r_t,sess1,model,value
	def loadingModel(self):
		
		Master.reading=1 
		print("IN AGENT")		
		self.model.load_weights(pathname+'\model.h5',by_name=True)		
		Master.reading=0	
	
		#------epsilon greedy action ------------------------#
	def chooseaction(self,current_state):

		prob=np.random.uniform(low=0,high=1)
		prob=1
		if prob<self.epsilon:
			return np.random.randint(low=0,high=self.action_num),0
		else:
			
			current_state=np.reshape(current_state,[-1,4])
			
			p,v=self.model.predict(current_state)
			
			action=np.argmax(p[0])
			return action,v	

	def run(self):

		current_state=np.zeros((1,4))
		total_reward=0
		steps=0
		for i in range(self.num_epsiodes):
			current_state=self.worker_env.reset()
			done=False
			while not done:
				steps+=1
				action,value=self.chooseaction(current_state)
				next_state,reward,done,_=self.worker_env.step(action)				
				total_reward+=reward
				
				action_one_hot=np.eye(self.action_num)[action]
				self.memory.append((current_state,action_one_hot,reward,next_state,value,done))	
				
				if done:
					#------------------------adding the experience to the trainingQ of global model-------#
					while len(self.memory)>0:
						n=len(self.memory)
						current_state,action,discounted_return,next_state,value,done1=self.calculateReturn(n)
						self.mastObj.trainingQ.append((current_state,action,discounted_return,next_state,value,done))
						self.memory.pop()
				


						
				if len(self.memory)>=self.n_step_return:
					current_state,action,discounted_return,next_state,value,done=self.calculateReturn(self.n_step_return)
					self.mastObj.trainingQ.append((current_state,action,discounted_return,next_state,value,done))			
				current_state=next_state	
				time.sleep(0.1)
				#-------decay epsilon----------#
				self.epsilon = Worker.MIN_EPSILON + (Worker.MAX_EPSILON - Worker.MIN_EPSILON) * math.exp(-Worker.LAMBDA * steps)
				print('Agent:=',self.agent_no,'Epsidoe',i,'Total reward=',total_reward)
				total_reward=0
				
				self.loadingModel()				
			
			self.mastObj.optimize=True
		self.mastObj.optimize=False
		self.mastObj.agent_count+=1
	
	#------------used to calculate the nstep reward function-------------------#
	def calculateReturn(self,n):
		
		for i in range(n):
			discounted_return=self.memory[i][2]*(self.gamma**i)
		current_state,action,reward,_,value,done=self.memory[0]
		_,_,_,next_state,_,_=self.memory[n-1]
		return current_state,action,discounted_return,next_state,value,done


	

		
