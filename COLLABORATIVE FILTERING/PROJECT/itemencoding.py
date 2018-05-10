import numpy as np
import re

itemProf = open('movies.dat')
ItemEncoding = np.zeros([3952,18])

for line in itemProf.readlines():
	x = re.split("[#\\n]+",line)
	y = [t for t in x[2].split('|')]
	#print(x[0])
	for genere in y:
		#print(genere)
		if(genere =='Action'):
			ItemEncoding[int(int(x[0])-1),0] = 1
		if(genere =='Adventure'):
			ItemEncoding[int(int(x[0])-1),1] = 1
		if(genere =='Animation'):
			ItemEncoding[int(int(x[0])-1),2] = 1
		if(genere =="Children's"):
			ItemEncoding[int(int(x[0])-1),3] = 1
		if(genere =='Comedy'):
			ItemEncoding[int(int(x[0])-1),4] = 1
		if(genere =='Crime'):
			ItemEncoding[int(int(x[0])-1),5] = 1
		if(genere =='Documentary'):
			ItemEncoding[int(int(x[0])-1),6] = 1
		if(genere =='Drama'):
			ItemEncoding[int(int(x[0])-1),7] = 1
		if(genere =='Fantasy'):
			ItemEncoding[int(int(x[0])-1),8] = 1
		if(genere =='Film-Noir'):
			ItemEncoding[int(int(x[0])-1),9] = 1
		if(genere =='Horror'):
			ItemEncoding[int(int(x[0])-1),10] = 1
		if(genere =='Musical'):
			ItemEncoding[int(int(x[0])-1),11] = 1
		if(genere =='Mystery'):
			ItemEncoding[int(int(x[0])-1),12] = 1
		if(genere =='Romance'):
			ItemEncoding[int(int(x[0])-1),13] = 1
		if(genere =='Sci-Fi'):
			ItemEncoding[int(int(x[0])-1),14] = 1
		if(genere =='Thriller'):
			ItemEncoding[int(int(x[0])-1),15] = 1
		if(genere =='War'):
			ItemEncoding[int(int(x[0])-1),16] = 1
		if(genere =='Western'):
			ItemEncoding[int(int(x[0])-1),17] = 1
	#print(ItemEncoding[int(int(x[0])-1),:])
