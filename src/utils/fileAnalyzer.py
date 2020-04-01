# -*- coding: utf-8 -*-
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
import numpy as np
import os

def countDistance(lat1,lon1,lat2,lon2):
    
    if min(lat1,lat2) < 38.5:
        return 0
    
    R = 6373000 #earth radius in meters

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    #print( distance)
    return distance

def plot_losses(history):
    hist = pd.DataFrame(history)
    plt.figure(figsize=(20,5))
    x = [i for i in range(len(hist))]
    for colnm in hist.columns:
        plt.plot(x,hist[colnm],label=colnm)
    plt.legend()
    plt.ylabel("Avg step (m)")
    plt.xlabel("Epoch")
    plt.grid()
    plt.show()
    #plt.savefig('lossOverEpochs.png')

#dir_data = "Trajectories/release/prediction_256/"
#dir_data = "Trajectories/release/matrices4/"
dir_data = "Trajectories/release/prediction_marcelove_kern4_256_kern15/"
#dir_data = "movementLogs/"    

history = []

"""
try all generated files and create plot:
    average distance of points per file generated in given epoch
    average number of huge steps (more than 150m per minute)
analysis of distance of generated nodes from real nodes on maps is performed in fileToMap.py (real trajectory generator)
"""

for epoch in range(0,1):
    total = 0
    numOfLargeSteps = 0
    numOfFilesPerEpoch = 10
    counter = 0
    ep = 0
    fileLen = 249
    
    names = np.sort(os.listdir(dir_data))
    for file in names:
        matrix = pd.read_csv(dir_data + file,delimiter='\t')#
        matrix = matrix.drop(matrix.columns[0], axis=1)
        matrix = matrix.values
        matrix = np.nan_to_num(matrix)
        
        oldlat = matrix[0,0]
        oldlon = matrix[0,1]
        total = 0
        for index in range(1,fileLen):
            lat = matrix[index,0]
            lon = matrix[index,1]
            length = countDistance(oldlat, oldlon, lat, lon)
            total += length
            if length>150: #large step is 150 meters per minute (9km/h - fast run //should be diff?//)
                numOfLargeSteps += 1
            #print("step % d is %d long for point one at coords %.5f, %.5f and second point %.5f, %.5f"%(index,length, oldlat, oldlon, lat, lon))
            oldlat = lat
            oldlon = lon
        counter += 1
        if counter == numOfFilesPerEpoch:
            counter = 0
            #○average step shouldnt exceed 25 meters per minute
            if total/(numOfFilesPerEpoch*fileLen)<26 and numOfLargeSteps/numOfFilesPerEpoch<20 and total/(numOfFilesPerEpoch*fileLen)>8:
                print("Average step in epoch %d was %d and number of huge steps was %d."%(ep, total/(numOfFilesPerEpoch*fileLen),numOfLargeSteps/numOfFilesPerEpoch))
            history.append({"Average":total/(numOfFilesPerEpoch*fileLen),"Big steps":numOfLargeSteps/numOfFilesPerEpoch})
            ep+=1
            total = 0
            numOfLargeSteps = 0
    

plot_losses(history)