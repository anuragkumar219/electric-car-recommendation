import pandas as pd
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
df=pd.read_csv('driver-data.csv')
print(df.head())
x=df.iloc[:,[1,2]].values
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
abc=[]
for i in range(2,10):
    model=KMeans(n_clusters=i)
    model.fit(x)
    abc.append(model.inertia_)
plt.plot(range(2,10),abc)
plt.xlabel('number of clusters')
plt.ylabel('within cluster of square sum')
plt.title('elbow method')
plt.show()
model=KMeans(n_clusters=6,random_state=0)
model.fit(x)
y_pred=model.predict(x)
y_pred
a=float(input('Enter distance travelled per day:'))
b=float(input('Enter speed value:'))
c=model.predict([[a,b]])
if(c==0):
    print('CLUSTER 0')
elif(c==1):
    print('CLUSTER 1')
elif(c==2):
    print('CLUSTER 2')
elif(c==3):
    print('CLUSTER 3')
elif(c==4):
    print('CLUSTER 4')
else:
    print('CLUSTER 5')
#scatter plot for first cluster
plt.scatter(x[y_pred==0,0],x[y_pred==0,1],label='cluster 0',c='r')
#scatter plot for second cluster
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],label='cluster 1',c='k')
#scatter plot for third cluster
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],label='cluster 2',c='b')
#scatter plot for fourth cluster
plt.scatter(x[y_pred==3,0],x[y_pred==3,1],label='cluster 3',c='c')
#scatter plot for fifth cluster
plt.scatter(x[y_pred==4,0],x[y_pred==4,1],label='cluster 4',c='g')
#scatter plot for sixth cluster
plt.scatter(x[y_pred==5,0],x[y_pred==5,1],label='cluster 5',c='#FFF800')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],c='y',s=200,label='centroid')
plt.legend()
plt.xlabel('Distance travelled per day')
plt.ylabel('average speed')
plt.show()

if(c==0):
    data_path = 'Cluster 0/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    print(onlyfiles[0])
    for i in onlyfiles:
     img = cv2.imread(i)
     cv2.imshow('recomended cars', img)
     cv2.waitKey(0)
     continue
elif(c==1):
    data_path = 'Cluster 1/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    print(onlyfiles[0])
    for i in onlyfiles:
     img = cv2.imread(i)
     cv2.imshow('recomended cars', img)
     cv2.waitKey(0)
     continue
elif(c==2):
    data_path = 'Cluster 2/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    print(onlyfiles[0])
    for i in onlyfiles:
        img = cv2.imread(i)
        cv2.imshow('recomended cars', img)
        cv2.waitKey(0)
        continue
elif(c==3):
    data_path = 'Cluster 3/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    print(onlyfiles[0])
    for i in onlyfiles:
        img = cv2.imread(i)
        cv2.imshow('recomended cars', img)
        cv2.waitKey(0)
        continue
elif(c==4):
    data_path = 'Cluster 4/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    print(onlyfiles[0])
    for i in onlyfiles:
        img = cv2.imread(i)
        cv2.imshow('recomended cars', img)
        cv2.waitKey(0)
        continue
else:
    data_path = 'Cluster 5/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    print(onlyfiles[0])
    for i in onlyfiles:
        img = cv2.imread(i)
        cv2.imshow('recomended cars', img)
        cv2.waitKey(0)
        continue



