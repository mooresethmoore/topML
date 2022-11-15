import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing.dummy import Pool

b1thresh = [-25, 50]
outLifetimeThresh = 1
life=int(b1thresh[1] - b1thresh[0]+1)
tBins = int(((life) ** 2 + life) / 2)

inDir="Z:/data/diverse_metals/diverseTopCSV/tThresh0/B1"
#topTypes=os.listdir(inDir)

outDir="Z:/data/diverse_metals/diverseTopPHvec/tThresh0/B1"

df=os.listdir(inDir)


def vectorizePH(phCSV, mi, ma, thresh=1, outLifetimeThresh=1):
    b1thresh = [round(mi), round(ma)]
    life=int(b1thresh[1] - b1thresh[0]+1)
    tBins = int(((life) ** 2 + life) / 2) #[(-mi,-mi)...,(-mi,ma),...(ma,ma)]
    PHvec = np.zeros(tBins + 4, dtype=np.int32)

    def indexMap(b, d):
        life = ma - mi +1 # technically b1thresh but let's keep it this way for now & assert type(mi,ma) ==(int,int)
        rows = round((b - mi))
        persist=d - b
        if rows < 0:
            if persist > thresh:
                return 0  # belowPersistent
            else:
                return 2  # belowWeak
        elif rows > life:
            if persist > thresh:
                return 1  # abovePersistent
            else:
                return 3  # aboveWeak
        else:
            if d>ma:
                if persist > thresh:
                    return 1 # abovePersistent ---- maybe we should separate into those abovePersistent
                        #with (b<0,b>=0) respectively? --looking at error cases it seems we're lucky and don't have
                        ### to deal with it for now (good bounds)

                        ####Remember, we can always decode the frequency plot (for resolution within bounds)
                else:
                    return 3 # aboveWeak
            rowskip = life - rows
            return round(4 + (life * (life + 1) - rowskip * (rowskip + 1)) / 2 + d - b)

    with open(phCSV, "r") as f:  ###GPU Parallelize here
        for line in f.readlines():
            b, d = line.split(",")
            PHvec[indexMap(np.float32(b), np.float32(d))] += 1
            #try:
                #PHvec[indexMap(np.float32(b), np.float32(d))] += 1
            #except:
               # print(f"\nERROR : {phCSV}\t:{b},{d}\t\tindexMap: {indexMap(np.float32(b), np.float32(d))}")
    return PHvec


def extractVec(mofName):
    PHvec=vectorizePH(f"{inDir}/{mofName}",b1thresh[0],b1thresh[1])
    np.save(f"{outDir}/{mofName[:-3]}npy",PHvec)

ncpus=6
pool=Pool(ncpus)


results=pool.map(extractVec,df)

pool.close()
pool.join()