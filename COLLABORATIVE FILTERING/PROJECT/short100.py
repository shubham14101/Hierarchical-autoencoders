import numpy as np
import tensorflow as tf
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



inputSize = 943
learning_rate = 0.005
logs_path = 'D:\loss'
lambdaR = 0.02
hiddenLayer1 = 120
hiddenLayer2 = 60
hiddenLayer3 = 30



mapping = tf.placeholder("float", [1682,inputSize]) 

X = tf.placeholder("float", [1682,inputSize])

V1 = tf.Variable(tf.random_uniform([inputSize ,hiddenLayer1],-1.0 / math.sqrt(inputSize ),1.0 / math.sqrt(inputSize )),trainable=True)
V2 = tf.Variable(tf.random_uniform([hiddenLayer1 ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
V3 = tf.Variable(tf.random_uniform([hiddenLayer2 ,hiddenLayer3],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
mu1 = tf.Variable(tf.zeros([hiddenLayer1]),trainable=True)
mu2 = tf.Variable(tf.zeros([hiddenLayer2]),trainable=True)
mu3 = tf.Variable(tf.zeros([hiddenLayer3]),trainable=True)
W3 = tf.Variable(tf.random_uniform([hiddenLayer3 ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer3 ),1.0 / math.sqrt(hiddenLayer3 )),trainable=True)
W2 = tf.Variable(tf.random_uniform([hiddenLayer2 ,hiddenLayer1],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
W1 = tf.Variable(tf.random_uniform([hiddenLayer1 ,inputSize],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
b3 = tf.Variable(tf.zeros([hiddenLayer2]),trainable=True)
b2 = tf.Variable(tf.zeros([hiddenLayer1]),trainable=True)
b1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
S1 = tf.Variable(tf.random_uniform([hiddenLayer1 ,inputSize],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
S2 = tf.Variable(tf.random_uniform([hiddenLayer2 ,inputSize],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
S3 = tf.Variable(tf.random_uniform([hiddenLayer3 ,inputSize],-1.0 / math.sqrt(hiddenLayer3 ),1.0 / math.sqrt(hiddenLayer3 )),trainable=True)
pi1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi2 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi3 = tf.Variable(tf.zeros([inputSize]),trainable=True)

E1 = tf.nn.sigmoid(tf.matmul(X,V1) + mu1)
E1drop = tf.nn.dropout(E1,0.5)
E2 = tf.nn.sigmoid(tf.add(tf.matmul(E1drop,V2),mu2))
E2drop = tf.nn.dropout(E2,0.5)
E3 = tf.nn.sigmoid(tf.add(tf.matmul(E2drop,V3),mu3))
E3drop = tf.nn.dropout(E3,0.5)
YS1 = tf.multiply(tf.identity(tf.add(tf.matmul(E1drop,S1),pi1)),mapping)
YS2 = tf.multiply(tf.identity(tf.add(tf.matmul(E2drop,S2),pi2)),mapping)
YS3 = tf.multiply(tf.identity(tf.add(tf.matmul(E3drop,S3),pi3)),mapping)
Ypool = (YS1 + YS2 + YS3)/3


regularize = layers.apply_regularization(layers.l2_regularizer(scale=lambdaR),weights_list=[V1,V2,V3,S1,S2,S3])


difference1NM = X - YS1
difference2NM = X - YS2
difference3NM = X - YS3
differencePool = X - Ypool


Loss1NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference1NM), 1, keep_dims=True))
Loss2NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference2NM), 1, keep_dims=True))
Loss3NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference3NM), 1, keep_dims=True))
LossPool = tf.reduce_mean(tf.reduce_sum(tf.square(differencePool), 1, keep_dims=True))


loss = Loss1NM + Loss2NM + Loss3NM + LossPool+ regularize

optimizer = layers.optimize_loss(loss=loss,global_step=tf.contrib.framework.get_global_step(),learning_rate=learning_rate,optimizer=tf.train.AdamOptimizer,summaries=["learning_rate","loss","gradients","gradient_norm",])



tf.summary.scalar("loss",loss)
merged_summary_op = tf.summary.merge_all()

MAE = np.zeros([5,5])
RMSE = np.zeros([5,5])
for i in range(0,5):
	trainIndex = np.array(random.sample(range(0,100000),90000))
	index = np.array(range(0,100000))
	testIndex = np.setdiff1d(index,trainIndex)
	UserMatrix = np.zeros([943,1682])
	ItemMatrix = np.zeros([1682,943])
	UserCompletion = np.zeros([943,1682])
	ItemCompletion = np.zeros([1682,943])
	traininSet = open('u.data')
	cc = 0
	for line in traininSet.readlines():
		x = [int(t) for t in line.split()]
		if(cc in trainIndex):
			UserMatrix[x[0]-1,x[1]-1] = x[2]
			ItemMatrix[x[1]-1,x[0]-1] = x[2]
			UserCompletion[x[0]-1,x[1]-1] = 1
			ItemCompletion[x[1]-1,x[0]-1] = 1
		cc = cc + 1

	meanDat = np.sum(UserMatrix)/np.sum(UserCompletion)
	itemBias = np.sum(ItemMatrix,axis = 1)/np.sum(ItemCompletion,axis = 1)
	userBias = np.sum(UserMatrix,axis = 1)/np.sum(UserCompletion,axis = 1)
	
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		for k in range(0,1000):
			
			for q in range(0,1):
				Xinput = np.zeros([1682,943])
				Xinput[:,:] = ItemMatrix[q*1682:(q+1)*1682,:]
				A = np.zeros([1682,943])
				A[:,:] = ItemCompletion[q*1682:(q+1)*1682,:]
				opt,lossP,summary,V_1,V_2,V_3,mu_1,mu_2,mu_3,W_3,W_2,W_1,b_3,b_2,b_1,S_1,S_2,S_3,pi_3,pi_2,pi_1= sess.run([optimizer,loss,merged_summary_op,V1,V2,V3,mu1,mu2,mu3,W3,W2,W1,b3,b2,b1,S1,S2,S3,pi3,pi2,pi1],feed_dict={X:Xinput,mapping:A})
				summary_writer.add_summary(summary,  k*1 + q)
			print(lossP)

		E_1 = sigm(np.matmul(ItemMatrix,V_1) + mu_1)
		
		E_2 = sigm(np.matmul(E_1,V_2) + mu_2)
		
		E_3 = sigm(np.matmul(E_2,V_3) + mu_3)
		Y_S1 = np.matmul(E_1,S_1) + pi_1
		Y_S2 = np.matmul(E_2,S_2) + pi_2
		Y_S3 = np.matmul(E_3,S_3) + pi_3
		Y_pool_1 = (Y_S1 +Y_S2+Y_S3)/3
		Y_pool_3 = Y_S1
		TestingSet = open('u.data')
		cc2 = 0
		predictionGroundTruthArray1 = np.empty((0,4), float)
		predictionGroundTruthArray2 = np.empty((0,4), float)
		predictionGroundTruthArray3 = np.empty((0,4), float)
		predictionGroundTruthArray4 = np.empty((0,4), float)
		predictionGroundTruthArray5 = np.empty((0,4), float)
		for line in TestingSet.readlines():
			x = [int(p) for p in line.split()]
			if(cc2 in testIndex):
				prediction1 = Y_pool_1[x[1]-1,x[0]-1]# - meanDat + itemBias[x[1]-1] + userBias[x[0]-1]
				#prediction2 = Y_last[x[1]-1,x[0]-1]# - meanDat + itemBias[x[1]-1] + userBias[x[0]-1]
				prediction3 = Y_S1[x[1]-1,x[0]-1]# - meanDat + itemBias[x[1]-1] + userBias[x[0]-1]
				prediction4 = Y_S2[x[1]-1,x[0]-1]
				prediction5 = Y_S3[x[1]-1,x[0]-1]
				print(prediction5,prediction4,prediction3,prediction1,x[2])
				if(not(math.isnan(prediction1) or math.isnan(prediction4) or math.isnan(prediction3) or math.isnan(prediction5))):
					predictionGroundTruthArray1 = np.append(predictionGroundTruthArray1,np.array([[x[0]-1,x[1]-1,x[2],prediction1]]),axis = 0)
					#predictionGroundTruthArray2 = np.append(predictionGroundTruthArray2,np.array([[x[0]-1,x[1]-1,x[2],prediction2]]),axis = 0)
					predictionGroundTruthArray3 = np.append(predictionGroundTruthArray3,np.array([[x[0]-1,x[1]-1,x[2],prediction3]]),axis = 0)
					predictionGroundTruthArray4 = np.append(predictionGroundTruthArray4,np.array([[x[0]-1,x[1]-1,x[2],prediction4]]),axis = 0)
					predictionGroundTruthArray5 = np.append(predictionGroundTruthArray5,np.array([[x[0]-1,x[1]-1,x[2],prediction5]]),axis = 0)

			cc2 = cc2 + 1

		
		MAE[i,0] = np.mean(np.absolute(predictionGroundTruthArray1[:,3]-predictionGroundTruthArray1[:,2]))
		#MAE[i,1] = np.mean(np.absolute(predictionGroundTruthArray2[:,3]-predictionGroundTruthArray2[:,2]))
		MAE[i,2] = np.mean(np.absolute(predictionGroundTruthArray3[:,3]-predictionGroundTruthArray3[:,2]))
		MAE[i,3] = np.mean(np.absolute(predictionGroundTruthArray4[:,3]-predictionGroundTruthArray4[:,2]))
		MAE[i,4] = np.mean(np.absolute(predictionGroundTruthArray5[:,3]-predictionGroundTruthArray5[:,2]))
		RMSE[i,0] = np.sqrt(np.mean(np.square(predictionGroundTruthArray1[:,3]-predictionGroundTruthArray1[:,2])))
		#RMSE[i,1] = np.sqrt(np.mean(np.square(predictionGroundTruthArray2[:,3]-predictionGroundTruthArray2[:,2])))
		RMSE[i,2] = np.sqrt(np.mean(np.square(predictionGroundTruthArray3[:,3]-predictionGroundTruthArray3[:,2])))
		RMSE[i,3] = np.sqrt(np.mean(np.square(predictionGroundTruthArray4[:,3]-predictionGroundTruthArray4[:,2])))
		RMSE[i,4] = np.sqrt(np.mean(np.square(predictionGroundTruthArray5[:,3]-predictionGroundTruthArray5[:,2])))


print(MAE)
print(RMSE)
