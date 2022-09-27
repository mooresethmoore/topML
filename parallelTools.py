import os
import sys
import multiprocessing
from multiprocessing import Process


cpus=multiprocessing.cpu_count() - 1
ident=lambda x: x
funcType=type(ident)
def dirDistribute(func,dirr,cpus=cpus,filterDir=None):
    """
    parallelize func(file) call for all files in dirr
    
    """
    semaphore = multiprocessing.Semaphore(cpus)
    def task(semaphore=semaphore,f=func,fil="test.txt"):
        with semaphore:
            f(fil)
    if type(filterDir)==funcType:
        processes=[Process(target=task,args=(semaphore,func,dirr+"/"+fil)) for fil in os.listdir(dirr) if filterDir(fil)]
    else:
        processes=[Process(target=task,args=(semaphore,func,dirr+"/"+fil)) for fil in os.listdir(dirr)]

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

def datDistribute(func,dat,cpus=cpus,filterFunc=None):
    """
    parallelize func(obj) call for all obj in dat
    
    """
    semaphore = multiprocessing.Semaphore(cpus)
    def task(semaphore=semaphore,f=func,fil="test.txt"):
        with semaphore:
            f(fil)
    if type(filterFunc)==funcType:
        processes=[Process(target=task,args=(semaphore,func,fil)) for fil in dat if filterFunc(fil)]
    else:
        processes=[Process(target=task,args=(semaphore,func,fil)) for fil in dat]

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()