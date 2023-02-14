import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gudhi,gudhi.hera,gudhi.wasserstein,persim



class PDhash():
    def __init__(self,res=1,diags=None, maxHdim=2,persistThresh=0,mode='freq'):
        """upper bound resolution.
        In the case of sparce PD spaces, it may be useful to project a hash map of your dataset to the diagram space"""
        self.res=res
        self.maxD=maxHdim
        self.thresh=persistThresh
        self.bounds=[[np.inf,-np.inf] for b in range(maxHdim+1)]
        self.img={b:dict() for b in range(maxHdim+1)} # While this does impose extra time compared to np, it is ideal for map-reduce type parallelization
        self.mode=mode # instead of freq, there is the 'set' (or None) option, that maps img[b][pt] to a set of indices rather than frequencies (dict:intensity)




    def addDiagRpp(self,diag,index): ## note the index can be just an index number, or a numerical value
                                    ###although the numerical values (duplicate index) won't stack in the set
        """diag is {0:[(b,d),...],1: """
        if self.mode=="freq":
            for i in range(np.min([self.maxD+1,len(diag)])):
                for k in diag[i]:
                    if k[1]-k[0] >self.thresh:
                        pt=(round(k[0]/self.res)*self.res,round(k[1]/self.res)*self.res)
                        if pt[0]<self.bounds[i][0]:
                            self.bounds[i][0]=pt[0]
                        if pt[1]>self.bounds[i][1]:
                            self.bounds[i][1]=pt[1]
                        if pt in self.img[i]:
                            if index in self.img[i][pt]:
                                self.img[i][pt][index]+=1
                            else:
                                self.img[i][pt][index]=1

                        else:
                            self.img[i][pt]={index:1}
        else:
            for i in range(np.min([self.maxD + 1, len(diag)])):
                for k in diag[i]:
                    if k[1] - k[0] > self.thresh:
                        pt = (round(k[0] / self.res) * self.res, round(k[1] / self.res) * self.res)
                        if pt[0] < self.bounds[i][0]:
                            self.bounds[i][0] = pt[0]
                        if pt[1] > self.bounds[i][1]:
                            self.bounds[i][1] = pt[1]
                        if pt in self.img[i]:
                            if index in self.img[i][pt]:
                                self.img[i][pt].add(index)
                        else:
                            self.img[i][pt] = {index}

    def addDiagCubeRips(self,crispy,index):
        """diag is [[bi,b,d,bx,by,bz,dx,dy,dz],..] """
        pass

    def __getitem__(self, item):
        if type(item)==int and item<=self.maxD: #item is bi
            return self.img[item]
        else:
            #return {b:self.img[b][pt] for b in range(self.maxD) for pt in self.img[b].keys()}
            return {b:self.img[b][tuple(item)] for b in range(self.maxD) if tuple(item) in self.img[b]}

    def remapIndex(self,index2New: dict):
        if self.mode=="freq":
            for b in self.img.keys():
                for pt in self.img[b].keys():
                    self.img[b][pt]={index2New[k]:v for k,v in self.img[b][pt].items()}
        else:
            for b in self.img.keys():
                for pt in self.img[b].keys():
                    self.img[b][pt]={index2New[k] for k in self.img[b][pt]}

    #def indexImgMap(self,fn=lambda pt: np.sum([v*k for k,v in self.img])): #assuming index is something numerically useful like a property assocated to each PD

    def mean_to_numpy(self,bi=None):
        roundDig=8
        if type(bi)==int and bi<=self.maxD:
            mi,ma=self.bounds[bi]
            life=int((self.bounds[bi][1]-self.bounds[bi][0])/self.res + 1)
            meanBox=np.zeros((life,life),dtype='float32')

            for k,v in self.img[bi].items():
                if self.mode=='freq': #type(v)==dict: #mode freq
                    meanBox[int((ma-k[1])/self.res),int((k[0]-mi)/self.res)]=np.round(np.sum([t*p for p,t in v.items()])/np.sum([t for p,t in v.items()]),roundDig)
                else:
                    meanBox[int((ma-k[1])/self.res),int((k[0]-mi)/self.res)]=np.round(np.mean(list(v)),roundDig)
            return meanBox

