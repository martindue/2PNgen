import numpy as np
import scipy.optimize as opt
from .signalSloper import signalSloper
import time

lookup_table = np.empty((0, 3))
lookup_table_txt = np.empty((0, 1), dtype='str')

def genNoise(N,lbl,genFun, rms_std, magnitude=1, AR=1, rotAng=0):
    global lookup_table, lookup_table_txt

    #TODO: add assertions for input



    # helper functions 
    def RMSSTD_from_slope(N, slope, genFun, lbl):
        nIter = 1000
        RMSSTDs = np.zeros(nIter)
        for i in range(nIter):
            RMS_, STD_ = calcRMSSTD(genData(N,slope,genFun))
            RMSSTDs[i] = RMS_/STD_
        RMSSTD = np.mean(RMSSTDs)

        return RMSSTD
    # Find the required spectral slope for N and RMS_STD
    key = (N, rms_std, lbl)
    print('Looking for ' + str(key) + ' in lookup table')
    slope = []
    if len(lookup_table)!=0:
        qSlope = (lookup_table[:,0] == N) & ((lookup_table[:,1] == rms_std) & (lookup_table_txt[:,0] == lbl))
        slope = lookup_table[qSlope,2]
    if len(slope) == 0:
        start = -1 if rms_std > np.sqrt(2) else 1
        searchFun = lambda x: np.abs(RMSSTD_from_slope(N,x,genFun,lbl)-rms_std)
        time_start = time.time()
        slope = opt.fmin(func = searchFun,x0 = start)
        stop = time.time() - time_start
        print("found slope for rms_std = ",str(rms_std)," and N = ",str(N)," to be ",str(slope[0])," in ",str(stop)," seconds")
        lookup_table = np.vstack((lookup_table, [N, rms_std, slope[0]]))
        lookup_table_txt = np.vstack((lookup_table_txt, lbl))
        print('Added ' + str([N, rms_std, slope[0]]) + ' to lookup table')
    else:
        print('Slope ',str(slope),' found in lookup table')

    
    data = genData(N,slope,genFun)

    # Make anisotropic, rotate and scale
    if AR != 1:
        data[1,:] = data[1,:]*AR
        rotAngRadians = np.radians(rotAng)
        data = [np.cos(rotAngRadians), -np.sin(rotAngRadians)], [np.sin(rotAngRadians), np.cos(rotAngRadians)] * data

    # Scale the data to desired magnitude

    RMS, STD = calcRMSSTD(data)
    data = data * magnitude / np.hypot(RMS, STD)
    return data




        
def genData(N, slope, genFun):
    data = np.concatenate((genFun(N),genFun(N)),axis=0)
    data = signalSloper(data,slope)
    return data

def calcRMSSTD(data):
    RMS = np.sqrt(np.mean(np.sum(np.diff(data, axis=1)**2, axis=0)))
    STD = np.sqrt(np.var(data[0,:]) +  np.var(data[1,:]))
    return RMS, STD