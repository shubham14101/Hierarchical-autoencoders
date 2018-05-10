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


male = np.array([0,1])
female = np.array([1,0])
classE = np.array([[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0]])
age1 = np.array([0,0,0,0,0,0,0,1])
age2 = np.array([0,0,0,0,0,0,1,0])
age3 = np.array([0,0,0,0,0,1,0,0])
age4 = np.array([0,0,0,0,1,0,0,0])
age5 = np.array([0,0,0,1,0,0,0,0])
age6 = np.array([0,0,1,0,0,0,0,0])
age7 = np.array([0,1,0,0,0,0,0,0])
age8 = np.array([1,0,0,0,0,0,0,0])
ocupation1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
ocupation2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
ocupation3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
ocupation4 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
ocupation5 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
ocupation6 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
ocupation7 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
ocupation8 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
ocupation9 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
ocupation10 = np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
ocupation11 = np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
ocupation12 = np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
ocupation13 = np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
ocupation14 = np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
ocupation15 = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
ocupation16 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
ocupation17 = np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
ocupation18 = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
ocupation19 = np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
ocupation20 = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
ocupation21 = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

UserItemMergeVector = np.zeros([1,50])
UserEncoding = np.zeros([943,31])
ItemEncoding = np.zeros([1682,19])
userProf = open('u.user')
itemProf = open('u.item')
for line in userProf.readlines():
	x = [t for t in line.split('|')]
	if(int(x[1])<14 and int(x[1])>=7):
		UserEncoding[int(x[0])-1,0:8] = age1
	elif(int(x[1])<=21 and int(x[1])>=14):
		UserEncoding[int(x[0])-1,0:8] = age2
	elif(int(x[1])<=28 and int(x[1])>=22):
		UserEncoding[int(x[0])-1,0:8] = age3
	elif(int(x[1])<=36 and int(x[1])>=29):
		UserEncoding[int(x[0])-1,0:8] = age4
	elif(int(x[1])<=48 and int(x[1])>=37):
		UserEncoding[int(x[0])-1,0:8] = age5
	elif(int(x[1])<=56 and int(x[1])>=49):
		UserEncoding[int(x[0])-1,0:8] = age6
	elif(int(x[1])<=65 and int(x[1])>=57):
		UserEncoding[int(x[0])-1,0:8] = age7
	elif(int(x[1])<=73 and int(x[1])>=66):
		UserEncoding[int(x[0])-1,0:8] = age8
	if(x[2] == 'M'):
		UserEncoding[int(x[0])-1,8:10] = male
	elif(x[2] == 'F'):
		UserEncoding[int(x[0])-1,8:10] = female
	if(x[3] == 'administrator'):
		UserEncoding[int(x[0])-1,10:31] = ocupation1
	elif(x[3] == 'artist'):
		UserEncoding[int(x[0])-1,10:31] = ocupation2
	elif(x[3] == 'doctor'):
		UserEncoding[int(x[0])-1,10:31] = ocupation3
	elif(x[3] == 'educator'):
		UserEncoding[int(x[0])-1,10:31] = ocupation4
	elif(x[3] == 'engineer'):
		UserEncoding[int(x[0])-1,10:31] = ocupation5
	elif(x[3] == 'entertainment'):
		UserEncoding[int(x[0])-1,10:31] = ocupation6
	elif(x[3] == 'executive'):
		UserEncoding[int(x[0])-1,10:31] = ocupation7
	elif(x[3] == 'healthcare'):
		UserEncoding[int(x[0])-1,10:31] = ocupation8
	elif(x[3] == 'homemaker'):
		UserEncoding[int(x[0])-1,10:31] = ocupation9
	elif(x[3] == 'lawyer'):
		UserEncoding[int(x[0])-1,10:31] = ocupation10
	elif(x[3] == 'librarian'):
		UserEncoding[int(x[0])-1,10:31] = ocupation11
	elif(x[3] == 'marketing'):
		UserEncoding[int(x[0])-1,10:31] = ocupation12
	elif(x[3] == 'none'):
		UserEncoding[int(x[0])-1,10:31] = ocupation13
	elif(x[3] == 'other'):
		UserEncoding[int(x[0])-1,10:31] = ocupation14
	elif(x[3] == 'programmer'):
		UserEncoding[int(x[0])-1,10:31] = ocupation15
	elif(x[3] == 'retired'):
		UserEncoding[int(x[0])-1,10:31] = ocupation16
	elif(x[3] == 'salesman'):
		UserEncoding[int(x[0])-1,10:31] = ocupation17
	elif(x[3] == 'scientist'):
		UserEncoding[int(x[0])-1,10:31] = ocupation18
	elif(x[3] == 'student'):
		UserEncoding[int(x[0])-1,10:31] = ocupation19
	elif(x[3] == 'technician'):
		UserEncoding[int(x[0])-1,10:31] = ocupation20
	elif(x[3] == 'writer'):
		UserEncoding[int(x[0])-1,10:31] = ocupation21

for line in itemProf.readlines():
	x = re.split("[|\\n]+",line)
	ItemEncoding[int(x[0])-1,0:19] =  list(map(int, x[4:23]))


inputSize = 943
learning_rate = 0.005
logs_path = 'D:\loss'
lambdaR = 0.02
hiddenLayer1 = 120
hiddenLayer2 = 60
hiddenLayer3 = 20



mapping = tf.placeholder("float", [1682,inputSize]) 

X = tf.placeholder("float", [1682,inputSize])
ItemSide = tf.placeholder("float", [1682,19])

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




Loss1NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference1NM), 1, keep_dims=True))
Loss2NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference2NM), 1, keep_dims=True))
Loss3NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference3NM), 1, keep_dims=True))
LossPool = tf.reduce_mean(tf.reduce_sum(tf.square(differencePool), 1, keep_dims=True))

#Loss1NM = tf.reduce_sum(tf.square(difference1NM))
#Loss2NM = tf.reduce_sum(tf.square(difference2NM))
#Loss3NM = tf.reduce_sum(tf.square(difference3NM))
#LossPool = tf.reduce_sum(tf.square(differencePool))

#Loss1NM = tf.reduce_mean(tf.nn.l2_loss(difference1NM))
#Loss2NM = tf.reduce_mean(tf.nn.l2_loss(difference2NM))
#Loss3NM = tf.reduce_mean(tf.nn.l2_loss(difference3NM))
#LossPool = tf.reduce_mean(tf.nn.l2_loss(differencePool))

loss = Loss1NM + Loss2NM + Loss3NM + LossPool+ regularize

optimizer = layers.optimize_loss(loss=loss,global_step=tf.train.get_global_step(),learning_rate=learning_rate,optimizer=tf.train.AdamOptimizer,summaries=["learning_rate","loss","gradients","gradient_norm",])


tf.summary.scalar("loss",loss)
merged_summary_op = tf.summary.merge_all()

MAE = np.zeros([10,1])
RMSE = np.zeros([10,1])
PRECISION = np.zeros([10,1])
RECALL = np.zeros([10,1])
for i in range(0,10):
	trainIndex = np.array(random.sample(range(0,100000),30000))
	index = np.array(range(0,100000))
	testIndex = np.setdiff1d(index,trainIndex)
	UserMatrix = np.zeros([943,1682])
	ItemMatrix = np.zeros([1682,943])
	UserCompletion = np.zeros([943,1682])
	ItemCompletion = np.zeros([1682,943])
	PredictionMatrix = np.zeros([1682,943])
	PredictionCompletion = np.zeros([1682,943])
	traininSet = open('u.data')
	cc = 0
	for line in traininSet.readlines():
		x = [int(t) for t in line.split()]
		if(cc in trainIndex):
			UserMatrix[x[0]-1,x[1]-1] = x[2]
			ItemMatrix[x[1]-1,x[0]-1] = x[2]
			UserCompletion[x[0]-1,x[1]-1] = 1
			ItemCompletion[x[1]-1,x[0]-1] = 1
		else:
			PredictionMatrix[x[1]-1,x[0]-1] = x[2]
			PredictionCompletion[x[1]-1,x[0]-1] = 1
		cc = cc + 1

	meanDat = np.sum(UserMatrix)/np.sum(UserCompletion)
	itemBias = np.sum(ItemMatrix,axis = 1)/np.sum(ItemCompletion,axis = 1)
	userBias = np.sum(UserMatrix,axis = 1)/np.sum(UserCompletion,axis = 1)
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

	
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		#inputLastIND = np.random.randint(2, size=(58,943))
		for k in range(0,700):
			
			for q in range(0,1):
				Xinput = np.zeros([1682,943])
				Xinput[:,:] = ItemMatrix[q*1682:(q+1)*1682,:]
				#Xnm = np.multiply(Xinput,inputLastIND)
				#Xm = Xinput - Xnm
				A = np.zeros([1682,943])
				A[:,:] = ItemCompletion[q*1682:(q+1)*1682,:]
				itemInfo = ItemEncoding[q*1682:(q+1)*1682,:]
				#Anm = np.multiply(A,inputLastIND)
				#Am = A - Anm
				opt,lossP,summary,V_1,V_2,V_3,mu_1,mu_2,mu_3,W_3,W_2,W_1,b_3,b_2,b_1,S_1,S_2,S_3,pi_3,pi_2,pi_1= sess.run([optimizer,loss,merged_summary_op,V1,V2,V3,mu1,mu2,mu3,W3,W2,W1,b3,b2,b1,S1,S2,S3,pi3,pi2,pi1],feed_dict={X:Xinput,mapping:A,ItemSide:itemInfo})
				summary_writer.add_summary(summary,  k*1 + q)
			print(lossP)

		newInp = np.array(np.concatenate((ItemMatrix,ItemEncoding),axis = 1))
		E_1 = sigm(np.matmul(ItemMatrix,V_1) + mu_1)
		#E_1 = np.array(np.concatenate((E_1,ItemEncoding),axis = 1))
		E_2 = sigm(np.matmul(E_1,V_2) + mu_2)
		#E_2 = np.array(np.concatenate((E_2,ItemEncoding),axis = 1))
		E_3 = sigm(np.matmul(E_2,V_3) + mu_3)
		#E_3 = np.array(np.concatenate((E_3,ItemEncoding),axis = 1))
		#D_2 = sigm(np.matmul(E_3,W_3) + b_3)
		#D_2 = np.array(np.concatenate((D_2,ItemEncoding),axis = 1))
		#D_1 = sigm(np.matmul(D_2,W_2) + b_2)
		#D_1 = np.array(np.concatenate((D_1,ItemEncoding),axis = 1))
		#Y_last = np.matmul(D_1,W_1) + b_1
		Y_S1 = np.matmul(E_1,S_1) + pi_1
		Y_S2 = np.matmul(E_2,S_2) + pi_2
		Y_S3 = np.matmul(E_3,S_3) + pi_3
		Y_pool_1 = (Y_S1 +Y_S2+Y_S3)/3
		#Y_pool_2 = Y_last
		Y_pool_3 = Y_S1
		p = np.multiply(Y_pool_1,PredictionCompletion)
		MAE[i,0] = np.sum(np.absolute(PredictionMatrix - p))/np.sum(PredictionCompletion)
		RMSE[i,0] = np.sqrt(np.sum(np.square(PredictionMatrix - p))/np.sum(PredictionCompletion))
		binPRed = PredictionMatrix
		binPRed[binPRed<3] = 0
		binPRed[binPRed > 3] = 1
		binP = p
		binP[binP<3] = 0
		binP[binP > 3] = 1
		tp = 0
		fp = 0
		fn = 0
		for elem1 in range(0,1682):
			for elem2 in range(0,943):
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
		for line in TestingSet.readlines():
			x = [int(p) for p in line.split()]
			if(cc2%10==i):
				prediction1 = Y_pool_1[x[1]-1,x[0]-1]# - meanDat + itemBias[x[1]-1] + userBias[x[0]-1]
				#prediction2 = Y_last[x[1]-1,x[0]-1]# - meanDat + itemBias[x[1]-1] + userBias[x[0]-1]
				prediction3 = Y_S1[x[1]-1,x[0]-1]# - meanDat + itemBias[x[1]-1] + userBias[x[0]-1]
				prediction4 = Y_S2[x[1]-1,x[0]-1]
				prediction5 = Y_S3[x[1]-1,x[0]-1]
				#print(prediction5,prediction4,prediction3,prediction1,x[2])
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
		'''


print(MAE)
print(RMSE)
print(PRECISION)
print(RECALL)
print(np.sum(MAE)/10,np.sum(RMSE)/10,np.sum(PRECISION)/10,np.sum(RECALL)/10)
#print(Y_pool_1)
#print(Y_S3)
#print(Y_S2)
#print(Y_S1)
#print(Y_last)