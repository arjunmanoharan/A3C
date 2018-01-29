import tensorflow as tf
import numpy as np
import threading
from collections import deque
import random
from keras.models import *
from keras.layers import *
from keras import backend as K


class Master(threading.Thread):
	trainingQ=deque(maxlen=100)
	optimize=True
	agent_count=0
	lock=threading.Lock()
	reading=0
	pathname=os.path.dirname(sys.argv[0])			
	def __init__(self,threadNo):
		threading.Thread.__init__(self)
		self.epsilon=1.0
		
		self.action_num=2	
		self.batch_size=3
		self.a_t=tf.placeholder('float32',(None,2))		
		self.r_t=tf.placeholder('float32',(None,1))
		self.inputs=tf.placeholder('float32',(None,4))
		self.num_agent=1
		self.gamma=0.95
		if threadNo==0:
			self.opt,self.loss_total,self.inputs,self.a_t,self.r_t,self.sess,self.model,self.value=self.buildModel()	
			self.save()
			print("##############global network#######################")
			

			
	
	
	def buildModel(self):
		new_graph=tf.Graph()
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

	def save(self):
		#-----------------saves the model--------------------#
		print("IN MASTER")
		
		while Master.reading==1: continue
		
		self.model.save_weights(pathname+'\model.h5')	
		
	
	def run(self):
		#------------code for fitting the global network---------------------#
		
		current_array=np.zeros((self.batch_size,4))
		reward_array=np.zeros((self.batch_size,1))
		action_array=np.zeros((self.batch_size,2))
		done_array=np.zeros((self.batch_size,1),dtype='bool')
		
		while Master.agent_count!=self.num_agent:
			print('optimizing global')		

			while Master.optimize:
				if self.batch_size<=len(Master.trainingQ):
					trainQ=random.sample(Master.trainingQ,self.batch_size)	
					for i in range(self.batch_size):
						
						current_array[i,:]=trainQ[i][0]
						reward_array[i,:]=trainQ[i][2]
						action_array[i,:]=trainQ[i][1]
						done_array[i,:]=trainQ[i][5]
					
					temp=self.sess.run([self.opt],feed_dict={self.inputs:current_array})#--------optimizing the network----------#
					
					q_current=reward_array+self.gamma*temp[0]
					q_current[done_array]=0
					
					self.save()#-------saving the optimized model-------------------#
		
	


	
