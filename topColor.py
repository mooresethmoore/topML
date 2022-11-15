import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing.dummy import Pool
from functools import reduce
import pickle

df=pd.read_csv(r"Z:\data\diverse_metals\post-combustion-vsa-2-clean.csv",index_col=0)
phDF=pd.read_csv("Z:/data/diverse_metals/phDF_tThresh0_B1.csv",index_col=0)

f=open("Z:/data/diverse_metals/phMOFmap-25_50.pkl","rb")
phMOFmap=pickle.load(f)
f.close()


channels=3
colorCodes={'max':(255,0,0),'min':(0,0,255)}


b0thresh=[-25,50]
mi,ma=b0thresh
life=ma-mi
meanBox=np.zeros((life+1,life+1,channels),dtype='uint8')

varBox=np.zeros((life+1,life+1,channels),dtype='uint8')

optVar="mmol/g_working_capacity"
boundOpt=[np.min(df[optVar]),np.max(df[optVar])]

def arrayImageIndexMap(b,d,mi,ma):
    return [ma-d,b-mi]


def colorBox(k): ## ie k = "50_50" the key to the associating region in bounding box-
    mi=-25;ma=50

    b,d=[int(i) for i in k.split("_")]
    subset = np.array([df.loc[k][optVar] for k in phMOFmap[k]],dtype=np.float64)
    if len(subset)>0:
        #inds=arrayImageIndexMap(b,d,mi,ma)
        meanBox[ma-d,b-mi,0]=np.mean(subset)
        if len(subset)>1:
            varBox[ma-d,b-mi,0]=np.var(subset)






ncpus=6
pool=Pool(ncpus)


results=pool.map(extractVec,df)

pool.close()
pool.join()