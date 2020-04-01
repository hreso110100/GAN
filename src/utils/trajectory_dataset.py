# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
from geographiclib.geodesic import Geodesic

class Loader():
    
    def __init__(self, channels=2, window=256):
        self.dir_data = "./movementLogs"
        self.fileList = self.getList()
        self.size = len(self.fileList)
        self.window = window
        self.channels = channels
        self.max_lat = 48.7171252
        self.min_lat = 48.7027684
        self.max_lon = 21.2497423
        self.min_lon = 21.228062
        self.lat_dist = (self.max_lat - self.min_lat) #0.015
        self.lon_dist = (self.max_lon - self.min_lon) #0.021
    
    def getList(self):

        names = np.sort(os.listdir(self.dir_data))
        return names
    
    def getLen(self):
        return self.size
    
    def load_batch(self,batch_size):
        """
        load trajectories from given folder
        window represents length of the loaded data, how many logs should be contained throughout one sample
        """
        if batch_size == 1:
            batch_size = 100
        indexes = np.random.randint(low=0,high=self.size,size=batch_size)
        
        shape = (self.window, 1, self.channels) #3 if bearing is used
        
    
        for index in indexes:
            """Latitude interval:  [48.7027684, 48.7171252]
            Longitude interval:  [21.228062, 21.2497423]"""
            
            batch = []
            corr = []
            try:
                matrix = pd.read_csv(self.dir_data + "/" + self.fileList[index], header=None)
            except:
                print("Failed to load file "+self.fileList[index])
                continue
            matrix = matrix.drop(matrix.columns[0], axis=1)
            matrix = matrix.values
            matrix = matrix[:self.window,:]
            
            if self.channels==3:
                distances = self.get_distance(matrix)
            
            #scale the values according to the lat/lon intervals
            lat = (matrix[:,0]-48.7)*10
            matrix[:,0] = lat
            lon = (matrix[:,1]-21.2)*10
            matrix[:,1] = lon
            matrix = np.nan_to_num(matrix)
            
            #additional channel
            if self.channels==3:
                matrix = np.concatenate((matrix,distances),1)
            
            
            
            matrix = matrix.reshape(shape)
            batch.append(matrix)
            corr = self.getCorruptFiles(batch)
            yield np.array(batch), np.array(corr)
            
    def load_batch_vanilla(self,batch_size,cols):
        """
        load trajectories from given folder
        window represents length of the loaded data, how many logs should be contained throughout one sample
        """
        if batch_size == 1:
            batch_size = 100
        indexes = np.random.randint(low=0,high=self.size,size=batch_size)
        
        #shape = (self.window, 1, self.channels) #3 if bearing is used
        shape = (self.window, cols, self.channels)
    
        for index in indexes:
            """Latitude interval:  [48.7027684, 48.7171252]
            Longitude interval:  [21.228062, 21.2497423]"""
            
            batch = []
            try:
                matrix = pd.read_csv(self.dir_data + "/" + self.fileList[index], header=None)
            except:
                print("Failed to load file "+self.fileList[index])
                continue
            matrix = matrix.drop(matrix.columns[0], axis=1)
            matrix = matrix.values
            matrix = matrix[:self.window,:]
            #scale the values according to the lat/lon intervals
            
            if self.channels!=2:
                distances = self.get_distance(matrix)
            
            lat = (matrix[:,0]-48.7)*30
            matrix[:,0] = lat
            lon = (matrix[:,1]-21.22)*30
            matrix[:,1] = lon
            matrix = np.nan_to_num(matrix)
            
            #additional channel
            if self.channels!=2:
                matrix = np.concatenate((matrix,distances),1)
            
            matrix2 = matrix.reshape(shape)
            batch.append(matrix2)
            #print(np.array(batch))
            yield np.array(batch), matrix
            
    def load_batch_lstm(self,batch_size,cols=0):
        """
        load trajectories from given folder
        window represents length of the loaded data, how many logs should be contained throughout one sample
        """
        if batch_size == 1:
            batch_size = 100
        indexes = np.random.randint(low=0,high=self.size,size=batch_size)
        
        #shape = (self.window, 1, self.channels) #3 if bearing is used
        shape = (self.window, 1, 1)
    
        for index in indexes:
            """Latitude interval:  [48.7027684, 48.7171252]
            Longitude interval:  [21.228062, 21.2497423]"""
            
            batch = []
            try:
                matrix = pd.read_csv(self.dir_data + "/" + self.fileList[index], header=None)
            except:
                print("Failed to load file "+self.fileList[index])
                continue
            matrix = matrix.drop(matrix.columns[0], axis=1)
            matrix = matrix.values
            matrix = matrix[:self.window,:]
            #scale the values according to the lat/lon intervals
            
            lat = (matrix[:,0]-48.7)*10
            lon = (matrix[:,1]-21.2)*10
            if cols == 0:
                matrix = lat
            if cols== 1:
                matrix = lon
            matrix = np.nan_to_num(matrix)
            
            matrix = matrix.reshape(shape)
            batch.append(matrix)
            #print(np.array(batch))
            yield np.array(batch)
            
    def load_batch_lstm_1D(self,batch_size):
        """
        load trajectories from given folder
        window represents length of the loaded data, how many logs should be contained throughout one sample
        """
        if batch_size == 1:
            batch_size = 10
        indexes = np.random.randint(low=0,high=self.size,size=batch_size)
        
        #shape = (self.window, 1, self.channels) #3 if bearing is used
        shape = (self.window, 1)
    
        for index in indexes:
            """Latitude interval:  [48.7027684, 48.7171252]
            Longitude interval:  [21.228062, 21.2497423]"""
            
            batch = []
            try:
                matrix = pd.read_csv(self.dir_data + "/" + self.fileList[index], header=None)
            except:
                print("Failed to load file "+self.fileList[index])
                continue
            matrix = matrix.drop(matrix.columns[0], axis=1)
            matrix = matrix.values
            matrix = matrix[:self.window,:]
            #scale the values according to the lat/lon intervals
                        
            #print(matrix[:5])
            
            
            #calculate "percentage" - the partition on 100x100 grid
            lat_offset = (matrix[:,0] - self.min_lat) / self.lat_dist
            #print(matrix[:5,1])
            lon_offset = (matrix[:,1] - self.min_lon) / self.lon_dist
            #print(lon_offset[:5])
            
            #so we can change type during offset counting
            lat_offset *= 100
            lat_offset = lat_offset.astype(int)
            lon_offset *= 100
            
            
            offset = lat_offset.astype(int)*100+lon_offset.astype(int)
            offset = offset/10000
            
            matrix = np.nan_to_num(offset)
            
            matrix = matrix.reshape(shape)
            batch.append(matrix)
            #print(np.array(batch))
            yield np.array(batch)
    
    def getCorruptFiles(self, files):
        """
        introduce random corruptions within files in dataset
        """
        corrupt = []
        size = len(files)
        i = 0
        for index in range(size):
            file = np.copy(files[index])
            for _ in range(int(self.window/5)): #remove random 20% points
                index = np.random.randint(low=0,high=self.window)
                file[index,0,0] = 0
                file[index,0,1] = 0
            corrupt.append(file)
            i+=1
        return corrupt
    
    def plot_losses(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(10,5))
        for colnm in hist.columns:
            plt.plot(hist[colnm],label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()
        
    def countDistance(self, lat1, lon1, lat2, lon2):

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

        
    def save_generated_data(self, epoch, batch, files_B, files_A, fake_A): 
        
#        files_A = files_A.reshape(3,self.window,2)
#        fake_A = fake_A.reshape(3,self.window,2)
#        files_B = files_B.reshape(3,self.window,2)
        avg = 0
        #larges = 0
        #print(files_B)
        for i in range(1):
            df = pd.DataFrame()
            df[0] = files_B[:,0]/10 + 48.7
            df[1] = files_B[:,1]/10 + 21.2
            df.to_csv('Trajectories/release/pix2pix/'+str(epoch)+'_'+str(batch)+'_base.csv', sep='\t')
            
            df2 = pd.DataFrame()
            df2[0] = fake_A[:,0]/10 + 48.7
            df2[1] = fake_A[:,1]/10 + 21.2
            df2.to_csv('Trajectories/release/pix2pix/'+str(epoch)+'_'+str(batch)+'_gener.csv', sep='\t')
            
            df3 = pd.DataFrame()
            df3[0] = files_A[:,0]/10 + 48.7
            df3[1] = files_A[:,1]/10 + 21.2
            df3.to_csv('Trajectories/release/pix2pix/'+str(epoch)+'_'+str(batch)+'_true.csv', sep='\t')

            for i in range(self.window):
                avg+=self.countDistance(df3[0][i],df3[1][i],df2[0][i],df2[1][i])
             
        # return larges/10,avg/10
        avg = avg/(self.window)
        if avg>300:
            avg = 300
        return avg
    
    def save_generated_data_vanilla(self, epoch, batch, files, folder = "prediction_marcelove_custom"): 
        #print(files.shape)
        for i in range(1):
            index = 0
            df = pd.DataFrame()
            df[index] = files[:,0]/10 + 48.7
            index+=1
            df[index] = files[:,1]/10 + 21.2
            index+=1
            if self.channels != 2:
                df[index] = files[:,2]*300
                index+=1
            df[index] = self.get_distance(files[:,:2])*300
            df.to_csv('Trajectories/release/'+folder+'/'+str(epoch)+'_'+str(batch)+'.csv', sep='\t')
            
    def save_generated_data_lstm(self, epoch, batch, files, folder = "lstm", latlon = 0): 
        #print(files.shape)
        for i in range(1):
            index = 0
            df = pd.DataFrame()
            if latlon==0:
                df[index] = files[:,0]/10 + 48.7
            else:
                df[index] = files[:,1]/10 + 21.2
            index+=1
            #df[index] = self.get_distance(files[:,:2])*300
            df.to_csv('Trajectories/release/'+folder+'/'+str(epoch)+'_'+str(batch)+'.csv', sep='\t')
            
    def save_generated_data_lstm_1D(self, epoch, batch, files, folder = "lstm_1d"): 
        #print(files[0]) debugovat toto!!!
        df = pd.DataFrame()
        
        df[0] = self.min_lat + ((files[0]*10000 - ((files[0]*10000)%100))/10000)*self.lat_dist
        #print((((files[:]*10000)%100)/100))
        df[1] = self.min_lon + (((files[0]*10000)%100)/100)*self.lon_dist
                        
        #df[index] = self.get_distance(files[:,:2])*300
        #print(df[:20])
        df.to_csv('Trajectories/release/'+folder+'/'+str(epoch)+'_'+str(batch)+'.csv', sep='\t')
            
    
    def get_bearing(self, file):
        lat1 = file[0,0]
        lon1 = file[0,1]
        bearing = []
        bearing.append([0])
        for index in range(1,self.window):
            lat2 = file[index,0]
            lon2 = file[index,1]
            brng = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']
            lat1 = lat2
            lon1 = lon2
            brng += 180
            brng /= 360
            bearing.append([brng])
            #print(brng)
        return np.array(bearing)
    
    def get_distance(self,file):
        
        #if passing a generated file
        if file[0,0]<1:
            file[:,0] = file[:,0]/10 + 48.7
            file[:,1] = file[:,1]/10 + 21.2
        
        length = []
        length.append([0])
        for i in range(1,len(file)):
            dist = self.countDistance(file[i-1,0],file[i-1,1],file[i,0],file[i,1])
            length.append([dist])
        
        arr = np.where(np.array(length)>300, 300, np.array(length))
        return arr/300
    
    
#l = Loader(channels=1)
#data = l.load_batch_lstm_1D(1)
#l.save_generated_data_lstm_1D(1,1,data)