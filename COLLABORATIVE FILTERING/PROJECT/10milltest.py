import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *
import math
import re
import random



def sigm(NumpyMat):	
	NumpyMat = 1 / (1 + np.exp(-NumpyMat))

	return NumpyMat

def softmax(NumpyMat):
	e_x = np.exp(NumpyMat - np.max(NumpyMat))
	return e_x / e_x.sum()




inputSize = 10681
learning_rate = 0.0005
logs_path = 'D:\loss'
lambdaR = 0.03
hiddenLayer1 = 900
hiddenLayer2 = 450
hiddenLayer3 = 150



mapping = tf.placeholder("float", [None,inputSize]) 
#mappingNM = tf.placeholder("float", [58,inputSize]) 
#mappingM = tf.placeholder("float", [58,inputSize])
#Xtrue = tf.placeholder("float", [58,inputSize]) 
X = tf.placeholder("float", [None,inputSize])
#ItemSide = tf.placeholder("float", [10681,18])
#XnotMasked = tf.placeholder("float",[58,inputSize])
#Xmasked = tf.placeholder("float",[58,inputSize])
#Xnew = tf.concat([X,ItemSide],1)
V1 = tf.Variable(tf.random_uniform([inputSize ,hiddenLayer1],-1.0 / math.sqrt(inputSize ),1.0 / math.sqrt(inputSize )),trainable=True)
V2 = tf.Variable(tf.random_uniform([hiddenLayer1 ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
V3 = tf.Variable(tf.random_uniform([hiddenLayer2 ,hiddenLayer3],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
mu1 = tf.Variable(tf.zeros([hiddenLayer1]),trainable=True)
mu2 = tf.Variable(tf.zeros([hiddenLayer2]),trainable=True)
mu3 = tf.Variable(tf.zeros([hiddenLayer3]),trainable=True)
S1 = tf.Variable(tf.random_uniform([hiddenLayer1 ,inputSize],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
S2 = tf.Variable(tf.random_uniform([hiddenLayer2 ,inputSize],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
S3 = tf.Variable(tf.random_uniform([hiddenLayer3 ,inputSize],-1.0 / math.sqrt(hiddenLayer3 ),1.0 / math.sqrt(hiddenLayer3 )),trainable=True)
pi1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi2 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi3 = tf.Variable(tf.zeros([inputSize]),trainable=True)

#E1 = tf.nn.sigmoid(tf.matmul(X,V1) + mu1)
#E1 = tf.concat([E1,ItemSide],1)
#E2 = tf.nn.sigmoid(tf.add(tf.matmul(E1,V2),mu2))
#E2 = tf.concat([E2,ItemSide],1)
#E3 = tf.nn.sigmoid(tf.add(tf.matmul(E2,V3),mu3))
#E3 = tf.concat([E3,ItemSide],1)
#D2 = tf.nn.sigmoid(tf.add(tf.matmul(E3,W3),b3))
#D2 = tf.concat([D2,ItemSide],1)
#D1 = tf.nn.sigmoid(tf.add(tf.matmul(E2,W2),b2))
#D1 = tf.concat([D1,ItemSide],1)
#Ylast = tf.multiply(tf.identity(tf.add(tf.matmul(D1,W1),b1)),mapping)
#YS1 = tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),S1),pi1)),mapping)
#YS2 = tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),S2),pi2)),mapping)
#YS3 = tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),V3),mu3)),S3),pi3)),mapping)
#Ypool = (tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),S1),pi1)),mapping) + tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),S2),pi2)),mapping) + tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),V3),mu3)),S3),pi3)),mapping))/3


#YlastM = tf.multiply(tf.identity(tf.add(tf.matmul(D1,W1),b1)),mappingM)
#YS1M = tf.multiply(tf.identity(tf.add(tf.matmul(E1,S1),pi1)),mappingM)
#YS2M = tf.multiply(tf.identity(tf.add(tf.matmul(E2,S2),pi2)),mappingM)
#YS3M = tf.multiply(tf.identity(tf.add(tf.matmul(E3,S3),pi3)),mappingM)

#YlastNM = tf.multiply(tf.identity(tf.add(tf.matmul(D1,W1),b1)),mappingNM)
#YS1NM = tf.multiply(tf.identity(tf.add(tf.matmul(E1,S1),pi1)),mappingNM)
#YS2NM = tf.multiply(tf.identity(tf.add(tf.matmul(E2,S2),pi2)),mappingNM)
#YS3NM = tf.multiply(tf.identity(tf.add(tf.matmul(E3,S3),pi3)),mappingNM)

#regularize = 


#differenceLastM = Xtrue - YlastM
#difference1M = Xtrue - YS1M
#difference2M = Xtrue - YS2M
#difference3M = Xtrue - YS3M


#LossLastM = tf.reduce_mean(tf.reduce_sum(tf.square(differenceLastM), 1, keep_dims=True))
#Loss1M = tf.reduce_mean(tf.reduce_sum(tf.square(difference1M), 1, keep_dims=True))
#Loss2M = tf.reduce_mean(tf.reduce_sum(tf.square(difference2M), 1, keep_dims=True))
#Loss3M = tf.reduce_mean(tf.reduce_sum(tf.square(difference3M), 1, keep_dims=True))

#differenceLastNM = X - Ylast
#difference1NM = 
#difference2NM = 
#difference3NM = 
#differencePool = 


#LossLastNM = tf.reduce_mean(tf.reduce_sum(tf.square(differenceLastNM), 1, keep_dims=True))
#Loss1NM = 
#Loss2NM = 
#Loss3NM = 
#LossPool = 


loss = tf.reduce_mean(tf.reduce_sum(tf.square(X - tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),S1),pi1)),mapping)), 1, keep_dims=True)) + tf.reduce_mean(tf.reduce_sum(tf.square(X - tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),S2),pi2)),mapping)), 1, keep_dims=True)) + tf.reduce_mean(tf.reduce_sum(tf.square(X - tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),V3),mu3)),S3),pi3)),mapping)), 1, keep_dims=True)) + tf.reduce_mean(tf.reduce_sum(tf.square(X - (tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),S1),pi1)),mapping) + tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),S2),pi2)),mapping) + tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),V3),mu3)),S3),pi3)),mapping))/3), 1, keep_dims=True))+ layers.apply_regularization(layers.l2_regularizer(scale=lambdaR),weights_list=[V1,V2,V3,S1,S2,S3])

optimizer = layers.optimize_loss(loss=loss,global_step=tf.train.get_global_step(),learning_rate=learning_rate,optimizer=tf.train.AdamOptimizer,summaries=["learning_rate","loss","gradients","gradient_norm",])
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#saver = tf.train.Saver()

#tf.summary.scalar("loss",loss)
#merged_summary_op = tf.summary.merge_all()

MAE = np.zeros([10,5])
RMSE = np.zeros([10,5])
for i in range(0,1):
	#print('0')
	#trainIndex = np.array(random.sample(range(0,1000209),500000))
	#index = np.array(range(0,1000209))
	#testIndex = np.setdiff1d(index,trainIndex)
	UserMatrix = np.zeros([71567,10681])
	#ItemMatrix = np.zeros([10681,71567])
	UserCompletion = np.zeros([71567,10681])
	#ItemCompletion = np.zeros([10681,71567])
	traininSet = open('ratings10mill.dat')
	cc = 0
	for line in traininSet.readlines():
		x = [float(t) for t in line.split('::')]
		print(cc)
		if(x[1]>10681):
			x[1] = round(x[1]/10)
		if(cc%10!=i):
			UserMatrix[int(x[0])-1,int(x[1])-1] = x[2]
			#ItemMatrix[int(x[1])-1,int(x[0])-1] = x[2]
			UserCompletion[int(x[0])-1,int(x[1])-1] = 1
			#ItemCompletion[int(x[1])-1,int(x[0])-1] = 1
		cc = cc + 1

	#meanDat = np.sum(UserMatrix)/np.sum(UserCompletion)
	#itemBias = np.sum(ItemMatrix,axis = 1)/np.sum(ItemCompletion,axis = 1)
	#userBias = np.sum(UserMatrix,axis = 1)/np.sum(UserCompletion,axis = 1)
	'''
	traininSet = open('u.data')
	cc1 = 0
	for line in traininSet.readlines():
		x = [int(t) for t in line.split()]
		if(cc1%2!=i):
			UserMatrix[x[0]-1,x[1]-1] = x[2]# + meanDat - itemBias[x[1]-1] - userBias[x[0] - 1]
			ItemMatrix[x[1]-1,x[0]-1] = x[2]# + meanDat - itemBias[x[1]-1] - userBias[x[0] - 1]
			UserCompletion[x[0]-1,x[1]-1] = 1
			ItemCompletion[x[1]-1,x[0]-1] = 1
		cc1 = cc1 + 1
	'''

	print('1')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		#summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		#inputLastIND = np.random.randint(2, size=(58,71567))
		#print('2')
		for k in range(0,300):
			print(k)
			losst = 0
			for q in range(0,59):
				#itemInfo = ItemEncoding[q*10681:(q+1)*10681,:]
				#Anm = np.multiply(A,inputLastIND)
				#Am = A - Anm
				opt,lossP,V_1,V_2,V_3,mu_1,mu_2,mu_3,S_1,S_2,S_3,pi_3,pi_2,pi_1= sess.run([optimizer,loss,V1,V2,V3,mu1,mu2,mu3,S1,S2,S3,pi3,pi2,pi1],feed_dict={X:UserMatrix[q*1213:(q+1)*1213:],mapping:UserCompletion[q*1213:(q+1)*1213,:]})
				#summary_writer.add_summary(summary,  k*1 + q)
				losst = losst + lossP
			
			print(losst/59)
		
		Y_pool_1 = np.zeros([71567,10681])
		for p in range(0,1213):
			print(p)
			Y_pool_1[p*59:(p+1)*59,:] = (np.matmul(sigm(np.matmul(UserMatrix[p*59:(p+1)*59,:],V_1) + mu_1),S_1) + pi_1 +np.matmul(sigm(np.matmul(sigm(np.matmul(UserMatrix[p*59:(p+1)*59,:],V_1) + mu_1),V_2) + mu_2),S_2) + pi_2+np.matmul(sigm(np.matmul(sigm(np.matmul(sigm(np.matmul(UserMatrix[p*59:(p+1)*59,:],V_1) + mu_1),V_2) + mu_2),V_3) + mu_3),S_3) + pi_3)/3

		TestingSet = open('ratings10mill.dat')
		cc2 = 0
		predictionGroundTruthArray1 = np.empty((0,4), float)

		for line in TestingSet.readlines():
			x = [float(p) for p in line.split('::')]
			if(x[1]>10681):
				x[1] = round(x[1]/10)
			if(cc2%10==i):
				prediction1 = Y_pool_1[int(x[0])-1,int(x[1])-1]
				#prediction3 = Y_S1[int(x[0])-1,int(x[1])-1]
				#prediction4 = Y_S2[int(x[0])-1,int(x[1])-1]
				#prediction5 = Y_S3[int(x[0])-1,int(x[1])-1]
				print(prediction1,x[2])
				if(not(math.isnan(prediction1))):
					predictionGroundTruthArray1 = np.append(predictionGroundTruthArray1,np.array([[x[0]-1,x[1]-1,x[2],prediction1]]),axis = 0)
					#predictionGroundTruthArray2 = np.append(predictionGroundTruthArray2,np.array([[x[0]-1,x[1]-1,x[2],prediction2]]),axis = 0)
					#predictionGroundTruthArray3 = np.append(predictionGroundTruthArray3,np.array([[int(x[0])-1,int(x[1])-1,x[2],prediction3]]),axis = 0)
					#predictionGroundTruthArray4 = np.append(predictionGroundTruthArray4,np.array([[int(x[0])-1,int(x[1])-1,x[2],prediction4]]),axis = 0)
					#predictionGroundTruthArray5 = np.append(predictionGroundTruthArray5,np.array([[int(x[0])-1,int(x[1])-1,x[2],prediction5]]),axis = 0)

			cc2 = cc2 + 1

		
		MAE[i,0] = np.mean(np.absolute(predictionGroundTruthArray1[:,3]-predictionGroundTruthArray1[:,2]))
		#MAE[i,1] = np.mean(np.absolute(predictionGroundTruthArray2[:,3]-predictionGroundTruthArray2[:,2]))
		#MAE[i,2] = np.mean(np.absolute(predictionGroundTruthArray3[:,3]-predictionGroundTruthArray3[:,2]))
		#MAE[i,3] = np.mean(np.absolute(predictionGroundTruthArray4[:,3]-predictionGroundTruthArray4[:,2]))
		#MAE[i,4] = np.mean(np.absolute(predictionGroundTruthArray5[:,3]-predictionGroundTruthArray5[:,2]))
		RMSE[i,0] = np.sqrt(np.mean(np.square(predictionGroundTruthArray1[:,3]-predictionGroundTruthArray1[:,2])))
		#RMSE[i,1] = np.sqrt(np.mean(np.square(predictionGroundTruthArray2[:,3]-predictionGroundTruthArray2[:,2])))
		#RMSE[i,2] = np.sqrt(np.mean(np.square(predictionGroundTruthArray3[:,3]-predictionGroundTruthArray3[:,2])))
		#RMSE[i,3] = np.sqrt(np.mean(np.square(predictionGroundTruthArray4[:,3]-predictionGroundTruthArray4[:,2])))
		#RMSE[i,4] = np.sqrt(np.mean(np.square(predictionGroundTruthArray5[:,3]-predictionGroundTruthArray5[:,2])))


print(MAE)
print(RMSE)