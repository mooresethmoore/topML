import os
import sys
import typing
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gudhi,gudhi.hera,gudhi.wasserstein,persim

import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.offline as pyo

from collections.abc import Iterable

class PDhash():
    def __init__(self, res=1, maxHdim=2, persistThresh=0, diags=None,  mode='freq'):
        """upper bound resolution.
        In the case of sparce PD spaces, it may be useful to project a hash map of your dataset to the diagram space"""
        self.res = res
        self.maxD = maxHdim
        self.thresh = persistThresh
        self.bounds = [[np.inf, -np.inf] for b in range(maxHdim + 1)]
        self.img = {b: dict() for b in range(
            maxHdim + 1)}  # While this does impose extra time compared to np, it is ideal for map-reduce type parallelization
        self.mode = mode  # instead of freq, there is the 'set' (or None) option, that maps img[b][pt] to a set of indices rather than frequencies (dict:intensity)
        # consider storing index items
    def addDiagRpp(self, diag, index,ptsigFig=4):
        """diag is {0:[(b,d),...],1:
            or [[(b,d),(b,d)...] for b in range(maxB)]
            index is key associated with this persistence diagram

            """
        ###although the numerical values (duplicate index) won't stack in the set

        ##note the index can be just an index number, or a numerical value -- most importantly, below statement must be true for expected behavior, but we leave it out fornow
        # assert isinstance(index,typing.Hashable)
        if self.mode == "freq":
            for i in range(np.min([self.maxD + 1, len(diag)])):
                for k in diag[i]:
                    if k[1] - k[0] > self.thresh:
                        pt = (round(round(k[0] / self.res) * self.res,ptsigFig),round( round(k[1] / self.res) * self.res),ptsigFig)
                        if pt[0] < self.bounds[i][0]:
                            self.bounds[i][0] = pt[0]
                        if pt[1] > self.bounds[i][1]:
                            self.bounds[i][1] = pt[1]
                        if pt in self.img[i]:
                            if index in self.img[i][pt]:
                                self.img[i][pt][index] += 1
                            else:
                                self.img[i][pt][index] = 1

                        else:
                            self.img[i][pt] = {index: 1}
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

    def addDiagCubeRips(self, crispy, index):
        """diag is [[bi,b,d,bx,by,bz,dx,dy,dz],..] """
        pass

    def __getitem__(self, item):
        if type(item) == int and item <= self.maxD:  # item is bi
            return self.img[item]
        else:
            # return {b:self.img[b][pt] for b in range(self.maxD) for pt in self.img[b].keys()}
            return {b: self.img[b][tuple(item)] for b in range(self.maxD) if tuple(item) in self.img[b]}

    def remapIndex(self, index2New: dict):
        if self.mode == "freq":
            for b in self.img.keys():
                for pt in self.img[b].keys():
                    self.img[b][pt] = {index2New[k]: v for k, v in self.img[b][pt].items()}
        else:
            for b in self.img.keys():
                for pt in self.img[b].keys():
                    self.img[b][pt] = {index2New[k] for k in self.img[b][pt]}

    def subsetIndex(self, subfunc=lambda x: True,changeBounds=False):  # not to be confused with subsetBounds
        #example subfunc= lambda x: x in subIndices
        subPD = PDhash(res=self.res, diags=None, maxHdim=self.maxD, persistThresh=self.thresh, mode=self.mode)
        if self.mode == "freq":
            subPD.img = {b: {pt: {k: v for k, v in self.img[b][pt].items() if subfunc(k)} for pt in self.img[b].keys()} for
                         b in self.img.keys()} #leaves fragment set() objects as img[b][pt] values --
            subPD.img = {b: {pt: v for pt, v in subPD.img[b].items() if v != {}} for b in
                         self.img.keys()}
        else:
            subPD.img = {b: {pt: {k for k in self.img[b][pt] if subfunc(k)} for pt in self.img[b].keys()} for b in
                         self.img.keys()}
            subPD.img = {b: {pt: v for pt, v in subPD.img[b].items() if v != set()} for b in
                         self.img.keys()}
        if changeBounds:
            subPD.bounds=[[min([k[0] if b in subPD.img.keys() else 0 for k in subPD.img[b].keys()]),max([k[1] if b in subPD.img.keys() else 0 for k in subPD.img[b].keys()])] for b in range(subPD.maxD+1)]
        else:
            subPD.bounds=self.bounds
        return subPD



    def merge(self,pdStack):
        # first shared keys, then add any new keys
        assert pdStack.maxD == self.maxD
        assert pdStack.res == self.res
        if self.mode == 'freq' and pdStack.mode=='freq':
            for bi in range(self.maxD+1):
                for pt in set(self.img[bi].keys()) & set(pdStack.img[bi].keys()):
                    self.img[bi][pt].update({ind:self.img[bi][pt][ind]+pdStack.img[bi][pt][ind] for ind in (set(self.img[bi][pt].keys()) & set(pdStack.img[bi][pt].keys()))})
                    self.img[bi][pt].update({ind:pdStack.img[bi][pt][ind] for ind in set(pdStack.img[bi][pt].keys())-set(self.img[bi][pt].keys())})

                self.img[bi].update({pt:pdStack.img[bi][pt] for pt in set(pdStack.img[bi].keys())-set(self.img[bi].keys())})
        ## change bounds
        self.bounds=[[np.min((self.bounds[b][0],pdStack.bounds[b][0])),np.max((self.bounds[b][1],pdStack.bounds[b][1]))] for b in range(len(self.bounds))]

    def density_to_numpy(self, bi=None): #how many unique points
        if type(bi) == int and bi <= self.maxD:
            mi, ma = self.bounds[bi]
            if mi != np.inf and ma != -np.inf:
                life = int((self.bounds[bi][1] - self.bounds[bi][0]) / self.res + 1)
                densBox = np.zeros((life, life), dtype=np.uint32)
                for k, v in self.img[bi].items():
                    densBox[int((ma - k[1]) / self.res), int((k[0] - mi) / self.res)] = len(v)
                return densBox
            else:
                print(f"no points in {bi}")
        else:
            densBoxes = []
            for bi in range(self.maxD + 1):
                mi, ma = self.bounds[bi]
                if mi != np.inf and ma != -np.inf:
                    life = int((self.bounds[bi][1] - self.bounds[bi][0]) / self.res + 1)
                    densBox = np.zeros((life, life), dtype=np.uint32)
                    for k, v in self.img[bi].items():
                        densBox[int((ma - k[1]) / self.res), int((k[0] - mi) / self.res)] = len(v)
                    densBoxes.append(densBox)
                else:
                    print(f"no points in {bi}")
                    densBoxes.append([])
            return densBoxes

    def mean_to_numpy(self, bi=None):
        roundDig = 8
        if type(bi) == int and bi <= self.maxD:
            mi, ma = self.bounds[bi]
            if mi != np.inf and ma != -np.inf:
                life = int((self.bounds[bi][1] - self.bounds[bi][0]) / self.res + 1)
                meanBox = np.zeros((life, life), dtype='float32')

                if self.mode == 'freq':  # type(v)==dict: #mode freq
                    for k, v in self.img[bi].items():
                        meanBox[int((ma - k[1]) / self.res), int((k[0] - mi) / self.res)] = np.round(
                            np.sum([t * p for p, t in v.items()]) / np.sum([t for p, t in v.items()]), roundDig)
                else:
                    for k, v in self.img[bi].items():
                        meanBox[int((ma - k[1]) / self.res), int((k[0] - mi) / self.res)] = np.round(np.mean(list(v)),
                                                                                                     roundDig)
                return meanBox
        else:
            # meanBox=np.zeros((self.maxD+1,)) # throw all to same scale? or list of meanBoxes
            # can we do more efficient in the plot? probably right
            meanBoxes = []
            for bi in range(self.maxD + 1):
                mi, ma = self.bounds[bi]
                if mi != np.inf and ma != -np.inf:
                    life = int((self.bounds[bi][1] - self.bounds[bi][0]) / self.res + 1)
                    meanBox = np.zeros((life, life), dtype='float32')

                    if self.mode == 'freq':  # type(v)==dict: #mode freq
                        for k, v in self.img[bi].items():
                            meanBox[int((ma - k[1]) / self.res), int((k[0] - mi) / self.res)] = np.round(
                                np.sum([t * p for p, t in v.items()]) / np.sum([t for p, t in v.items()]), roundDig)
                    else:
                        for k, v in self.img[bi].items():
                            meanBox[int((ma - k[1]) / self.res), int((k[0] - mi) / self.res)] = np.round(
                                np.mean(list(v)), roundDig)
                    meanBoxes.append(meanBox)
                else:
                    print(f"no points in {bi}")
                    meanBoxes.append([])

            return meanBoxes

    def sum_to_numpy(self, bi=None):
        roundDig = 8
        if type(bi) == int and bi <= self.maxD:
            mi, ma = self.bounds[bi]
            if mi != np.inf and ma != -np.inf:
                life = int((self.bounds[bi][1] - self.bounds[bi][0]) / self.res + 1)
                meanBox = np.zeros((life, life), dtype='float32')

                if self.mode == 'freq':  # type(v)==dict: #mode freq
                    for k, v in self.img[bi].items():
                        meanBox[int((ma - k[1]) / self.res), int((k[0] - mi) / self.res)] = np.round(
                            np.sum([t * p for p, t in v.items()]), roundDig)
                else:# type(v)==set : #mode None or 'dict'
                    for k, v in self.img[bi].items():
                        meanBox[int((ma - k[1]) / self.res), int((k[0] - mi) / self.res)] = np.round(np.sum(list(v)),
                                                                                                     roundDig)
                return meanBox
        else:
            # meanBox=np.zeros((self.maxD+1,)) # throw all to same scale? or list of meanBoxes
            # can we do more efficient in the plot? probably right
            meanBoxes = []
            for bi in range(self.maxD + 1):
                mi, ma = self.bounds[bi]
                if mi != np.inf and ma != -np.inf:
                    life = int((self.bounds[bi][1] - self.bounds[bi][0]) / self.res + 1)
                    meanBox = np.zeros((life, life), dtype='float32')

                    if self.mode == 'freq':  # type(v)==dict: #mode freq
                        for k, v in self.img[bi].items():
                            meanBox[int((ma - k[1]) / self.res), int((k[0] - mi) / self.res)] = np.round(
                                np.sum([t * p for p, t in v.items()]), roundDig)
                    else:
                        for k, v in self.img[bi].items():
                            meanBox[int((ma - k[1]) / self.res), int((k[0] - mi) / self.res)] = np.round(
                                np.sum(list(v)), roundDig)
                    meanBoxes.append(meanBox)
                else:
                    print(f"no points in {bi}")
                    meanBoxes.append([])

            return meanBoxes

    def meanFigs(self,savePref=None,bounds=None,boxBoundMax=None,returnTraces=True,showFig=True,titSuf="",colormp="sunsetdark"):
        ht,wt=1080,1920

        traces=[]
        if isinstance(bounds, Iterable):
            if len(bounds)==2 and not isinstance(bounds[0], Iterable):
                bounds=[bounds for b in range(self.maxD+1)]#use 0 and 1 as consistentbounds
            elif len(bounds)!=self.maxD+1 or not (isinstance(bounds[0], Iterable) and len(bounds[0])!=2):
                bounds = self.bounds
        else:
            bounds = [np.min([self.bounds[b][0] for b in range(self.maxD+1)]),np.max([self.bounds[b][1] for b in range(self.maxD+1)])]#self.bounds # or use max
            bounds = [bounds for b in range(self.maxD + 1)]
        if not boxBoundMax:
            if self.mode=="freq":

                boxBoundMax=np.max(list(set.union(*(set(self.img[b][pt].keys()) for b in self.img.keys() for pt in self.img[b].keys()))))
            else:
                boxBoundMax = np.max(
                    list(set.union(*(self.img[b][pt] for b in self.img.keys() for pt in self.img[b].keys()))))
        for b in range(self.maxD+1):
            X = mean_to_numpy(self, bi=b)
            X[X == 0] = np.nan
            x = np.linspace(self.bounds[b][0], self.bounds[b][1], len(X)) # this line is bugged right now,
            y = x
            trace = go.Heatmap(x=x, y=y, z=X[::-1], colorscale=colormp, autocolorscale=False, zmax=boxBoundMax, zmin=0)
            traces.append(trace)
            if savePref:
                saveSuf=f"b{b}_Mean"
                layout = go.Layout(title=f"B{b} {titSuf}", height=ht, width=wt,
                                   xaxis=dict(range=[bounds[b][0], bounds[b][1]], ),
                                   yaxis=dict(range=[bounds[b][0], bounds[b][1]], ))
                fig=go.Figure(trace,layout=layout)
                pyo.plot(fig, filename=f"{savePref}{saveSuf}.html", auto_open=False)
                fig.write_image(f"{savePref}{saveSuf}.png")

        if savePref:
            saveSuf=f"bALL_Mean"
            fig = sp.make_subplots(rows=1, cols=self.maxD+1)  # go.Figure(data=[trace], layout=layout)
            for b in range(len(traces)):
                fig.add_trace(traces[b],row=1,col=b+1)
            layout=dict(title=f"{titSuf}", height=ht, width=wt,
                      xaxis=dict(range=[bounds[b][0], bounds[b][1]], ),
                      yaxis=dict(range=[bounds[b][0], bounds[b][1]], ))
            fig.update_layout(layout)
            pyo.plot(fig, filename=f"{savePref}{saveSuf}.html", auto_open=False)
            fig.write_image(f"{savePref}{saveSuf}.png")
            if showFig:
                fig.show()


        if returnTraces:
            return traces

    def npFigProj(self,bettiProj=lambda s,b: mean_to_numpy(s,b),savePref=None,bounds=None,boxBoundMax=None,returnTraces=True,showFig=True,titSuf="",colormp="sunsetdark"):
        ht,wt=1080,1920

        traces=[]
        if isinstance(bounds, Iterable):
            if len(bounds)==2 and not isinstance(bounds[0], Iterable):
                bounds=[bounds for b in range(self.maxD+1)]#use 0 and 1 as consistentbounds
            elif len(bounds)!=self.maxD+1 or not (isinstance(bounds[0], Iterable) and len(bounds[0])!=2):
                bounds = self.bounds
        else:
            bounds = [np.min([self.bounds[b][0] for b in range(self.maxD+1)]),np.max([self.bounds[b][1] for b in range(self.maxD+1)])]#self.bounds # or use max
            bounds = [bounds for b in range(self.maxD + 1)]
        if not boxBoundMax:
            if self.mode=="freq":

                boxBoundMax=np.max(list(set.union(*(set(self.img[b][pt].keys()) for b in self.img.keys() for pt in self.img[b].keys()))))
            else:
                boxBoundMax = np.max(
                    list(set.union(*(self.img[b][pt] for b in self.img.keys() for pt in self.img[b].keys()))))
        for b in range(self.maxD+1):
            X = bettiProj(self, b)# assuming bettiProj is one of the mean/density/sum functions in this class but can be any betti numpy projection
            #X[X == 0] = np.nan
            x = np.linspace(bounds[b][0], bounds[b][1], len(X))
            y = x
            trace = go.Heatmap(x=x, y=y, z=X[::-1], colorscale=colormp, autocolorscale=False, zmax=boxBoundMax, zmin=0)
            traces.append(trace)
            if savePref:
                saveSuf=f"b{b}_Mean"
                layout = go.Layout(title=f"B{b} {titSuf}", height=ht, width=wt,
                                   xaxis=dict(range=[bounds[b][0], bounds[b][1]], ),
                                   yaxis=dict(range=[bounds[b][0], bounds[b][1]], ))
                fig=go.Figure(trace,layout=layout)
                pyo.plot(fig, filename=f"{savePref}{saveSuf}.html", auto_open=False)
                fig.write_image(f"{savePref}{saveSuf}.png")

        if savePref:
            saveSuf=f"bALL_Mean"
            fig = sp.make_subplots(rows=1, cols=self.maxD+1)  # go.Figure(data=[trace], layout=layout)
            for b in range(len(traces)):
                fig.add_trace(traces[b],row=1,col=b+1)
            layout=dict(title=f"{titSuf}", height=ht, width=wt,
                      xaxis=dict(range=[bounds[b][0], bounds[b][1]], ),
                      yaxis=dict(range=[bounds[b][0], bounds[b][1]], ))
            fig.update_layout(layout)
            pyo.plot(fig, filename=f"{savePref}{saveSuf}.html", auto_open=False)
            fig.write_image(f"{savePref}{saveSuf}.png")
            if showFig:
                fig.show()


        if returnTraces:
            return traces


    def boxStatsIndex(self):
        return {b: {pt: (np.mean(np.array(list(self.img[b][pt]), dtype=np.float32)),
                         np.var(np.array(list(self.img[b][pt]), dtype=np.float32))) for pt in self.img[b].keys()} for b
                in self.img.keys()}



def boxStatsIndex(pdStack):
    return {b: {pt: (
    np.mean(np.array(list(pdStack[b][pt]), dtype=np.float32)), np.var(np.array(list(pdStack[b][pt]), dtype=np.float32)))
                for pt in pdStack[b].keys()} for b in pdStack.img.keys()}


def mean_to_numpy(pdStack, bi=None):
    roundDig = 8
    if type(bi) == int and bi <= pdStack.maxD:
        mi, ma = pdStack.bounds[bi]
        life = int((pdStack.bounds[bi][1] - pdStack.bounds[bi][0]) / pdStack.res + 1)
        meanBox = np.zeros((life, life), dtype='float32')

        for k, v in pdStack.img[bi].items():
            if type(v) == dict:
                meanBox[int((ma - k[1]) / pdStack.res), int((k[0] - mi) / pdStack.res)] = np.round(
                    np.sum([t * p for p, t in v.items()]) / np.sum([t for p, t in v.items()]), roundDig)
            else:
                meanBox[int((ma - k[1]) / pdStack.res), int((k[0] - mi) / pdStack.res)] = np.round(np.mean(list(v)),
                                                                                                   roundDig)
        return meanBox

def boxProjectSet(pdStack,df,regVar,indexMap=None):### set project vs freq project
    pdProj=PDhash(res=pdStack.res,diags=None,maxHdim=pdStack.maxD,persistThresh=pdStack.thresh)
    pdProj.bounds=pdStack.bounds
    pdProj.mode="set"
    if indexMap:
        pdProj.img={b:{pt:{df.loc[indexMap[i]][regVar] for i in pdStack.img[b][pt]} for pt in pdStack.img[b].keys()} for b in pdStack.img.keys()}
    else: #assume the index of the pdStack is referring to the numerical index of the df
        #this will return an error if indices in pdStack are higher than in df but this is expected
        # behavior to prevent unexpected behavior later on
        pdProj.img={b:{pt:{df.iloc[i][regVar] for i in pdStack.img[b][pt]} for pt in pdStack.img[b].keys()} for b in pdStack.img.keys()}
    return pdProj


def boxProjectFreq(pdStack,df,regVar,indexMap=None):### set project vs freq project
    assert pdStack.mode=="freq"
    pdProj=PDhash(res=pdStack.res,diags=None,maxHdim=pdStack.maxD,persistThresh=pdStack.thresh)
    pdProj.bounds=pdStack.bounds
    if indexMap:
        pdProj.img={b:{pt:{df.loc[indexMap[i]][regVar]:v for i,v in pdStack.img[b][pt].items()} for pt in pdStack.img[b].keys()} for b in pdStack.img.keys()}
    else: #assume the index of the pdStack is referring to the numerical index of the df
        #this will return an error if indices in pdStack are higher than in df but this is expected
        # behavior to prevent unexpected behavior later on
        pdProj.img={b:{pt:{df.iloc[i][regVar]:v for i,v in pdStack.img[b][pt].items()} for pt in pdStack.img[b].keys()} for b in pdStack.img.keys()}
    return pdProj