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



inputSize = 100
learning_rate = 0.002
logs_path = 'D:\loss'
lambdaR = 0.05
hiddenLayer1 = 30
hiddenLayer2 = 20
hiddenLayer3 = 10


mapping = tf.placeholder("float", [73421,inputSize]) 

X = tf.placeholder("float", [73421,inputSize])


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

E2 = tf.nn.sigmoid(tf.add(tf.matmul(E1,V2),mu2))

E3 = tf.nn.sigmoid(tf.add(tf.matmul(E2,V3),mu3))

YS1 = tf.multiply(tf.identity(tf.add(tf.matmul(E1,S1),pi1)),mapping)
YS2 = tf.multiply(tf.identity(tf.add(tf.matmul(E2,S2),pi2)),mapping)
YS3 = tf.multiply(tf.identity(tf.add(tf.matmul(E3,S3),pi3)),mapping)
Ypool = (YS1 + YS2 + YS3)/3


regularize = layers.apply_regularization(layers.l2_regularizer(scale=lambdaR),weights_list=[V1,V2,V3,S1,S2,S3])



difference1NM = X - YS1
difference2NM = X - YS2
difference3NM = X - YS3
differencePool = X - Ypool


#Loss1NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference1NM), 1, keep_dims=True))
#Loss2NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference2NM), 1, keep_dims=True))
#Loss3NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference3NM), 1, keep_dims=True))
#LossPool = tf.reduce_mean(tf.reduce_sum(tf.square(differencePool), 1, keep_dims=True))
Loss1NM = tf.reduce_sum(tf.square(difference1NM))
Loss2NM = tf.reduce_sum(tf.square(difference2NM))
Loss3NM = tf.reduce_sum(tf.square(difference3NM))
LossPool = tf.reduce_sum(tf.square(differencePool))

loss = Loss1NM + Loss2NM + Loss3NM + LossPool+ regularize

optimizer = layers.optimize_loss(loss=loss,global_step=tf.train.get_global_step(),learning_rate=learning_rate,optimizer=tf.train.AdamOptimizer,summaries=["learning_rate","loss","gradients","gradient_norm",])


tf.summary.scalar("loss",loss)
merged_summary_op = tf.summary.merge_all()

MAE = np.zeros([10,1])
RMSE = np.zeros([10,1])
PRECISION = np.zeros([10,1])
RECALL = np.zeros([10,1])
jesterData = open('jester.txt')
newData = np.zeros([7342100,3])
cc3 = 0
for line in jesterData.readlines():
	x = re.split("[,\\n]+",line)
	newData[cc3*100:(cc3+1)*100,0] = cc3
	newData[cc3*100:(cc3+1)*100,1] = np.array(range(0,100))
	newData[cc3*100:(cc3+1)*100,2] = list(map(float, x[0:100]))
	cc3 = cc3+1
#print(newData)
index = np.where(newData[:,2] == 99)[0]
newData = np.delete(newData,index,axis = 0)
newData[:,2] = newData[:,2]/10
print(newData)

for i in range(0,1):
	trainIndex = np.array(random.sample(range(0,4136360),3722724))
	indexT = np.array(range(0,100000))
	testIndex = np.setdiff1d(indexT,trainIndex)
	UserMatrix = np.zeros([73421,100])
	UserCompletion = np.zeros([73421,100])
	PredictionMatrix = np.zeros([73421,100])
	PredictionCompletion = np.zeros([73421,100])
	cc = 0
	for row in range(0,4136360):
		
		if(row%10!=i):
			print(row)
			UserMatrix[int(newData[row,0]),int(newData[row,1])] = newData[row,2]
			UserCompletion[int(newData[row,0]),int(newData[row,1])] = 1
		else:
			PredictionMatrix[int(newData[row,0]),int(newData[row,1])] = newData[row,2]
			PredictionCompletion[int(newData[row,0]),int(newData[row,1])] = 1
		cc = cc + 1



	

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		#inputLastIND = np.random.randint(2, size=(58,943))
		for k in range(0,1800):
			
			for q in range(0,1):
				
				opt,lossP,summary,V_1,V_2,V_3,mu_1,mu_2,mu_3,W_3,W_2,W_1,b_3,b_2,b_1,S_1,S_2,S_3,pi_3,pi_2,pi_1= sess.run([optimizer,loss,merged_summary_op,V1,V2,V3,mu1,mu2,mu3,W3,W2,W1,b3,b2,b1,S1,S2,S3,pi3,pi2,pi1],feed_dict={X:UserMatrix,mapping:UserCompletion})
				summary_writer.add_summary(summary,  k*1 + q)
			print(lossP)


		E_1 = sigm(np.matmul(UserMatrix,V_1) + mu_1)
		E_2 = sigm(np.matmul(E_1,V_2) + mu_2)
		E_3 = sigm(np.matmul(E_2,V_3) + mu_3)
		
		Y_S1 = np.matmul(E_1,S_1) + pi_1
		Y_S2 = np.matmul(E_2,S_2) + pi_2
		Y_S3 = np.matmul(E_3,S_3) + pi_3
		Y_pool_1 = (Y_S1 +Y_S2+Y_S3)/3

		Y_pool_3 = Y_S1
		p = np.multiply(Y_pool_1,PredictionCompletion)
		MAE[i,0] = np.sum(np.absolute(PredictionMatrix - p))/np.sum(PredictionCompletion)
		RMSE[i,0] = np.sqrt(np.sum(np.square(PredictionMatrix - p))/np.sum(PredictionCompletion))
		binPRed = PredictionMatrix
		binPRed[binPRed<0] = 0
		binPRed[binPRed >0] = 1
		binP = p
		binP[binP<0] = 0
		binP[binP > 0] = 1
		tp = 0
		fp = 0
		fn = 0
		for elem1 in range(0,73421):
			for elem2 in range(0,100):
				if(binP[elem1,elem2] == 1 and binP[elem1,elem2] == binPRed[elem1,elem2]):
					tp = tp +1
				elif(binP[elem1,elem2] == 1 and binP[elem1,elem2] != binPRed[elem1,elem2]):
					fp = fp +1
				elif(binP[elem1,elem2] == 0 and binP[elem1,elem2] != binPRed[elem1,elem2]):
					fn = fn +1
		preicison = tp/(tp+fp)
		recall = tp/(tp + fn)
		PRECISION[i,0] = preicison
		RECALL[i,0] = recall
		'''
		TestingSet = open('u.data')
		cc2 = 0
		predictionGroundTruthArray1 = np.empty((0,4), float)
		predictionGroundTruthArray2 = np.empty((0,4), float)
		predictionGroundTruthArray3 = np.empty((0,4), float)
		predictionGroundTruthArray4 = np.empty((0,4), float)
		predictionGroundTruthArray5 = np.empty((0,4), float)
		for row in range(0,4136360):
			if(cc2%10==i):
				#print(row)
				prediction1 = Y_pool_1[int(newData[row,0]),int(newData[row,1])]
				prediction3 = Y_S1[int(newData[row,0]),int(newData[row,1])]
				prediction4 = Y_S2[int(newData[row,0]),int(newData[row,1])]
				prediction5 = Y_S3[int(newData[row,0]),int(newData[row,1])]
				print(prediction5,prediction4,prediction3,prediction1,newData[row,2])
				if(not(math.isnan(prediction1) or math.isnan(prediction4) or math.isnan(prediction3) or math.isnan(prediction5))):
					predictionGroundTruthArray1 = np.append(predictionGroundTruthArray1,np.array([[int(newData[row,0]),int(newData[row,1]),newData[row,2],prediction1]]),axis = 0)
					predictionGroundTruthArray3 = np.append(predictionGroundTruthArray3,np.array([[int(newData[row,0]),int(newData[row,1]),newData[row,2],prediction3]]),axis = 0)
					predictionGroundTruthArray4 = np.append(predictionGroundTruthArray4,np.array([[int(newData[row,0]),int(newData[row,1]),newData[row,2],prediction4]]),axis = 0)
					predictionGroundTruthArray5 = np.append(predictionGroundTruthArray5,np.array([[int(newData[row,0]),int(newData[row,1]),newData[row,2],prediction5]]),axis = 0)

			cc2 = cc2 + 1

		
		MAE[i,0] = np.mean(np.absolute(predictionGroundTruthArray1[:,3]-predictionGroundTruthArray1[:,2]))
		MAE[i,2] = np.mean(np.absolute(predictionGroundTruthArray3[:,3]-predictionGroundTruthArray3[:,2]))
		MAE[i,3] = np.mean(np.absolute(predictionGroundTruthArray4[:,3]-predictionGroundTruthArray4[:,2]))
		MAE[i,4] = np.mean(np.absolute(predictionGroundTruthArray5[:,3]-predictionGroundTruthArray5[:,2]))
		RMSE[i,0] = np.sqrt(np.mean(np.square(predictionGroundTruthArray1[:,3]-predictionGroundTruthArray1[:,2])))
		RMSE[i,2] = np.sqrt(np.mean(np.square(predictionGroundTruthArray3[:,3]-predictionGroundTruthArray3[:,2])))
		RMSE[i,3] = np.sqrt(np.mean(np.square(predictionGroundTruthArray4[:,3]-predictionGroundTruthArray4[:,2])))
		RMSE[i,4] = np.sqrt(np.mean(np.square(predictionGroundTruthArray5[:,3]-predictionGroundTruthArray5[:,2])))
		'''

print(MAE)
print(RMSE)
print(PRECISION)
print(RECALL)
print(np.sum(MAE)/10,np.sum(RMSE)/10,np.sum(PRECISION)/10,np.sum(RECALL)/10)
