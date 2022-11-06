import os
import sys
import h5py
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import distance_transform_edt
import tcripser
import gudhi,gudhi.hera,gudhi.wasserstein,persim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ase
from ase.io import cube
from ase.io import cif
import multiprocessing
from multiprocessing.dummy import Pool
from joblib import Parallel, delayed

atHome=True

if atHome:
    df = pd.read_csv(r"Z:\data\diverse_metals\post-combustion-vsa-2-clean.csv")
    espDir = r"Z:\data\diverse_metals\ESPs"
    outDir= r"Z:\data\diverse_metals\diverseTop"
else:
    df = pd.read_csv("/lus/grand/projects/ACO2RDS/mooreseth/post-combustion-vsa-2-clean.csv")
    espDir = "/lus/grand/projects/ACO2RDS/DATA/diverse_metals\ESPs"
    outDir = r"/lus/grand/projects/ACO2RDS/mooreseth/diverseTop"


def lifeThresh(diags,eps=0):
    return [np.array([[j[0],j[1]] for j in diags[b] if not np.isinf(j[1]) and j[1]-j[0]>=eps]) for b in range(len(diags))]


def persistCube(cub,maxD=3,inf=False,outThresh=lambda x: x):
    """Remember crispy can also have locations corresponding to birth death cells, can identify regions near clusters
    There's also the --hist option to see a cell's participation for future imp"""
    crispy=tcripser.computePH(cub,maxdim=maxD)

    diags=[np.array([[j[1],j[2]] for j in crispy if j[0]==b]) for b in range(maxD)]
    if not inf:
        try:
            if (len(diags[0]))>1:
                diags[0]=diags[0][:-1]
        except:
            print("error")
    maxD=-1
    for i in range(len(diags)):
        if len(diags[i])>0:
            maxD=i

    return outThresh(diags),maxD

def geoDist(cub,eps):
    binCub = (cub >= eps) ##eps=threshold_otsu(cub)
    return distance_transform_edt(binCub)-distance_transform_edt(~binCub)

def persistGeoVoxel(cub,eps=0,maxD=2,inf=False,outThresh=lambda x: x):
    """uses EDT distance transform on binary image above eps """
    dt_img = geoDist(cub,eps)

    crispy=tcripser.computePH(dt_img,maxdim=maxD)

    diags=[np.array([[j[1],j[2]] for j in crispy if j[0]==b]) for b in range(maxD)]
    if not inf:
        try:
            if (len(diags[0]))>1:
                diags[0]=diags[0][:-1]
        except:
            print("error")
    maxD=-1
    for i in range(len(diags)):
        if len(diags[i])>0:
            maxD=i

    return outThresh(diags),maxD





def mofGrab(cubFile):#,maxD=2,inf=False,compression_opts=8):  # ,outThresh=lambda x: x): no outThresh for now for conv
    maxD=3
    inf=False
    compression_opts=8

    with h5py.File(f"{espDir}/{cubFile}.hdf5", "r") as f:
        cub=f["vdata"]["total"][()]
    ### func first
    crispy = tcripser.computePH(cub, maxdim=maxD-1)
    diags = np.array([np.array([np.array([j[1], j[2]],dtype=np.float64) for j in crispy if j[0] == b],dtype=object) for b in range(maxD)],dtype=object)
    if not inf:
        if (len(diags[0])) > 1:
            diags[0] = diags[0][:-1]

    np.save(f'{outDir}/tFunc/{cubFile}_PH.npy',diags)
    #with h5py.File(f'{outDir}/tFunc/{cubFile}PH.hdf5', 'w') as out:
    #    out.create_dataset('data',diags)#,compression='gzip',compression_opts=compression_opts)

    ### Now Geo
    eps=0
    cub=geoDist(cub,eps)

    crispy = tcripser.computePH(cub, maxdim=maxD-1)
    diags = np.array([np.array([np.array([j[1], j[2]],dtype=np.float64) for j in crispy if j[0] == b],dtype=object) for b in range(maxD)],dtype=object)
    if not inf:
        if (len(diags[0])) > 1:
            diags[0] = diags[0][:-1]

    np.save(f'{outDir}/tThresh0/{cubFile}_PH.npy', diags)
    #with h5py.File(f'{outDir}/tThresh0/{cubFile}PH.hdf5', 'w') as out:
    #    out.create_dataset('data', diags)#, compression='gzip', compression_opts=compression_opts)


    ### eps -.051
    eps = -.051
    cub = geoDist(cub, eps)

    crispy = tcripser.computePH(cub, maxdim=maxD-1)
    diags = np.array([np.array([np.array([j[1], j[2]],dtype=np.float64) for j in crispy if j[0] == b],dtype=object) for b in range(maxD)],dtype=object)
    if not inf:
        if (len(diags[0])) > 1:
            diags[0] = diags[0][:-1]

    np.save(f'{outDir}/tThresh-05/{cubFile}_PH.npy', diags)
    #with h5py.File(f'{outDir}/tThresh-05/{cubFile}PH.hdf5', 'w') as out:
    #    out.create_dataset('data', diags)#, compression='gzip', compression_opts=compression_opts)






ncpus=6
pool=Pool(ncpus)
results=pool.map(mofGrab,list(df["filename"]))

pool.close()
pool.join()



#Parallel(n_jobs=ncpus)(delayed(mofGrab)(i for i in df["filename"]))