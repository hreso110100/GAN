# -*- coding: utf-8 -*-

import pandas as pd
from src.city.Map import Map
import numpy as np
from src.common.Location import Location
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt


def countDistance(lat1,lon1,lat2,lon2):
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

def getMidPoint(maxlat,maxlon,minlat,minlon):    
    midlat = minlat + (maxlat-minlat)/2 #middle point
    midlon = minlon + (maxlon-minlon)/2
    return Location(midlat,midlon)

def plot_distances(history):
    hist = pd.DataFrame(history)
    plt.figure(figsize=(20,5))
    for colnm in hist.columns:
        plt.plot(hist[colnm],label=colnm)
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.show()
    
#create map just once 
maxlat = 48.72
maxlon = 21.25
minlat = 48.7
minlon = 21.227
mapLen = countDistance(maxlat,maxlon,minlat,minlon)/2
loca = getMidPoint(maxlat,maxlon,minlat,minlon)
mapka = Map(int(mapLen/2+100),False,loca)

def getPathInFile(filename):
    #create the map with given trajectory
    filepath = "Trajectories/release/prediction_marcelove_kern4_256/"+filename
    matrix = pd.read_csv(filepath,delimiter='\t')#
    matrix = matrix.drop(matrix.columns[0], axis=1)
    matrix = matrix.values
    matrix = np.nan_to_num(matrix)

    maxD, avgD = mapka.plotRoute2(matrix, "generated maps/"+filename, matrix.shape[0])
    
    #return the values to create average
    return (maxD, avgD)

history = []
for i in range(1,401):
    avg = 0
    maximal = 0
    for j in range(10):
        filename = str(i)+"_"+str(j)+".csv"
        maxD, avgD = getPathInFile(filename)
        if maxD>maximal:
            maximal = maxD
        avg += avgD
    history.append({"Average step":avg/10,"maximal step":maximal})
plot_distances(history)