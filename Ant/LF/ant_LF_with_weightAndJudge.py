from math import pow,sqrt,exp
import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import random

def _dis(_ds,_w,_i,_j):
    diff = _ds[_i]-_ds[_j]
    return sqrt(np.dot(_w,diff*diff))

sampleNum = 150 # sample totality
gridNum = 40 # grid num of each column and row
featureNum = 4 # feature totality
classNum = 5 # cluster totality
antNum = 15 # ant totality
iterNum = 10000 # maximum iterate totality

R = 2
K1 = 0.1
K2 = 0.15
alpha = 0.15
V = 1

#sample, target_classify = ds.load_iris(True)
sample, target_classify = ds.make_blobs(sampleNum, n_features=featureNum, centers=classNum, random_state=13) # gennerate data
#print(sample)

def dataPreparation(): # Normalization
    maxVal = np.zeros(featureNum)
    minVal = np.zeros(featureNum)
    for i in range(featureNum):
        maxVal[i] = np.max(sample[:,i])
        minVal[i] = np.min(sample[:,i])
    data = (sample - minVal) / np.abs(maxVal-minVal)
    return data

data = dataPreparation() # normalization
#print("data:\n"+repr(data))


weight = np.ones(featureNum) # weight of features
dis = np.zeros([sampleNum,sampleNum]) # Euclid dis between two samples

def weightPreparation():
    sum = 0
    featureSum = np.zeros(featureNum)
    for featureIdx in range(featureNum):
        for i in range(sampleNum):
            for j in range(sampleNum):
                dlt = abs(data[i,featureIdx] - data[j,featureIdx])
                featureSum[featureIdx] += dlt
                sum += dlt
    return featureSum / sum
#weight = weightPreparation()

def distancePreparation(): # calculate the feature distance
    for i in range(sampleNum):
        for j in range(i):
            dis[i,j] = _dis(data,weight,i,j)
            dis[j,i] = dis[i,j]

distancePreparation()


occpied = np.zeros([gridNum,gridNum]) # if there is a sample on [x,y]
pos = np.zeros([sampleNum,2]) # the pos of sample points
picked = np.zeros(sampleNum) # if the sample is loaded now

def projectSample(): # project samples randomly to the (gridNum*gridNum)-surface
    L = 0
    R = gridNum - 1
    for i in range(sampleNum):
        x = random.randint(L,R)
        y = random.randint(L,R)
        while (occpied[x,y]==1):
            x = random.randint(L,R)
            y = random.randint(L,R)
        occpied[x,y] = 1
        pos[i] = [x,y]

projectSample()

ant = -1 * np.ones(antNum)
antPos = np.zeros([antNum,2]) # the pos of ants

def projectAntToSample(antIdx):
    L = 0
    R = sampleNum - 1
    sampleIdx = random.randint(L,R)
    while (picked[sampleIdx] == 1):
        sampleIdx = random.randint(L,R)
    return sampleIdx

def gridDis(a,b):
    a = int(a)
    b = int(b)
    [x1,y1] = pos[a]
    [x2,y2] = pos[b]
    return max(int(abs(x1-x2)),int(abs(y1-y2)))

def similarity(sampleIdx):
    S = 2 * R + 1
    ret = 0
    sampleIdx = int(sampleIdx)
    for i in range(sampleNum):
        if (i == sampleIdx):
            continue
        elif (gridDis(i,sampleIdx)<=R):
            ret += 1 - dis[i,sampleIdx] / alpha
    ret /= S * S
    ret = max(0,ret)
    return ret

def pickUpProb(sim):
    return pow(K1 / (K1 + sim), 2)

def putDownProb(sim):
    if (sim<K2):
        return 2 * sim
    else:
        return 1

def pickUpProcess(antIdx,sampleIdx,alterRate):
    sim = similarity(sampleIdx)
    pickProb = pickUpProb(sim)
    if (pickProb <= alterRate):
        return False
    [x,y] = pos[sampleIdx]
    x = int(x)
    y = int(y)
    occpied[x,y] = 1
    picked[sampleIdx] = 1
    ant[antIdx] = sampleIdx
    antPos[antIdx] = [x,y]
    return True

def putDownProcess(antIdx,alterRate):
    sampleIdx = int(ant[antIdx])
    sim = similarity(sampleIdx)
    putProb = putDownProb(sim)
    if (putProb < alterRate):
        return -1
    [x,y] = antPos[antIdx]
    x = int(x)
    y = int(y)
    if (occpied[x,y] == 1):
        return 1
    occpied[x,y] = 1
    pos[sampleIdx] = [x,y]
    ant[antIdx] = -1
    picked[sampleIdx] = 0
    return 0
    
def projectAnt(antIdx,V):
    L = 0
    R = 3
    d = random.randint(L,R)
    dx = [1,0,-1,0]
    dy = [0,1,0,-1]
    [x,y] = antPos[antIdx]
    x += dx[d]*V
    y += dy[d]*V
    x = max(min(gridNum-1,x),0)
    y = max(min(gridNum-1,y),0)
    antPos[antIdx] = [x,y]
    sampleIdx = int(ant[antIdx])
    pos[sampleIdx] = [x,y]

plt.scatter(pos[:,0], pos[:,1], c=target_classify)
plt.show()

jdg = np.zeros([sampleNum])

iterCnt = 0
while (iterCnt <= iterNum):
    print(iterCnt)
    jdg = np.zeros([sampleNum])
    for antIdx in range(antNum):
        if (ant[antIdx] == -1): # unloaded now
            alterRate = random.random()
            sampleIdx = int(projectAntToSample(antIdx))
            while (jdg[sampleIdx] == 1):
                sampleIdx = projectAntToSample(antIdx)
            succPick = pickUpProcess(antIdx,sampleIdx,alterRate)
            while (succPick == False):
                sampleIdx = projectAntToSample(antIdx)
                alterRate = random.random()
                succPick = pickUpProcess(antIdx,sampleIdx,alterRate)
            jdg[sampleIdx] = 1
            projectAnt(antIdx,V)
        elif (ant[antIdx] != -1):
            projectAnt(antIdx,V)
            alterRate = random.random()
            t = putDownProcess(antIdx,alterRate)
            if (t!=0):
                projectAnt(antIdx,V)
            
    iterCnt += 1

fa = [i for i in range(sampleNum)]
def find(x):
    if (fa[x] == x):
        return x
    else:
        fa[x] = find(fa[x])
        return fa[x]
def checkResult():
    for i in range(sampleNum):
        for j in range(sampleNum):
            if (gridDis(i,j) <= R):
                faI = find(i)
                faJ = find(j)
                if (faI == faJ):
                    continue
                fa[faI] = fa[faJ]

    cluster = [list() for i in range(sampleNum)]
    for i in range(sampleNum):
        faI = find(i)
        cluster[faI].append(i)
    
    ctr = np.zeros([sampleNum,featureNum])
    ctrNum = 0
    for i in range(sampleNum):
        if (len(cluster[i]) == 0):
            continue
        tempC = np.zeros([featureNum])
        for j in cluster[i]:
            tempC += sample[j]
        tempC /= len(cluster[i])
        ctr[ctrNum] = tempC
        ctrNum += 1
    #print(repr(ctr[0:ctrNum]))


checkResult()

plt.scatter(pos[:,0], pos[:,1], c=target_classify)
plt.show()

plt.scatter(data[:, 1], data[:, 2], c=target_classify, s=20)
plt.show() 




