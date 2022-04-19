# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:22:39 2020

@author: user
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def density_function(path, file_num):
    pd.options.display.max_colwidth = 12000
    par = pd.read_csv('/'.join([path, file_num]), encoding='euc-kr')
    raw_data = par.loc[80].to_string(index=False)
    raw_data = raw_data.replace(':',';').split(';')
    raw_data = raw_data[1:1025]
    raw_data_re = np.array(raw_data).astype(float)
    raw_data_re = raw_data_re.reshape(32,32)[::-1]
    
    particle_diameter = np.array([0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187, 1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 
                                  3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500, 9.500, 11.000, 13.000, 15.000, 17.000, 19.000, 21.500, 24.500])
    particle_speed = np.array([0.05, 0.150, 0.250, 0.350, 0.450, 0.550, 0.650, 0.750, 0.850, 0.950, 1.100, 1.300, 1.500, 1.700, 1.900, 2.200, 2.600, 3.000, 
                               3.400, 3.800, 4.400, 5.200, 6.000, 6.800, 7.600, 8.800, 10.400, 12.000, 13.600, 15.200, 17.600, 20.800])
    particle_speed = particle_speed.reshape(32,1)
    particle_volume = 4/3*np.pi*(particle_diameter/2*0.001)**3
    particle_volume = particle_volume.reshape(1,32)
    
    vol_speed_class = (particle_speed*particle_volume)[::-1]
    
    vol_speed_density = vol_speed_class * raw_data_re
    
    sum_vol_speed_density = []
    for i in range(32):
        sum_vol_speed_density_raw = sum(vol_speed_density[:,i])
        sum_vol_speed_density.append(float(sum_vol_speed_density_raw))
        
    sum_vol_speed_density = np.array(sum_vol_speed_density)
    density_function = sum_vol_speed_density/np.sum(sum_vol_speed_density)
    
    return density_function, sum_vol_speed_density, raw_data_re

def pars_db(path):
    file_list=os.listdir(path)        
    pd.options.display.max_colwidth = 12000
    particle_diameter = [0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187, 1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 
                                  3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500, 9.500, 11.000, 13.000, 15.000, 17.000, 19.000, 21.500, 24.500]
    
    db_raw_data_rain_drop = pd.DataFrame([])    
    db_raw_data_rain_intensity = pd.DataFrame([])
    db_date_time = pd.DataFrame([])
    
    for i in range(len(file_list)):
        par = pd.read_csv('/'.join([path, file_list[i]]), encoding='euc-kr')
        
        # particle size distribution
        raw_data_rain_drop = par.loc[78].to_string(index=False)
        raw_data_rain_drop = raw_data_rain_drop.replace(':',';').split(';')
        raw_data_rain_drop = raw_data_rain_drop[1:33]
        
        # rain intensity
        raw_data_rain_intensity = float(par.columns.tolist()[0].split(':')[1])
        
        # date
        time = np.array(par.loc[18].to_string(index=False).split(':')).astype(int)
        date = np.array(par.loc[19].to_string(index=False).replace('.',':').split(':')).astype(int)
        date_time = datetime(date[3],date[2],date[1],time[1],time[2],time[3])
        
        db_raw_data_rain_drop = db_raw_data_rain_drop.append([raw_data_rain_drop], ignore_index = True)
        db_raw_data_rain_intensity = db_raw_data_rain_intensity.append(pd.Series([raw_data_rain_intensity]), ignore_index = True)
        db_date_time = db_date_time.append([date_time], ignore_index = True)
    
    db_raw_data_rain_drop.columns = particle_diameter
    db_raw_data_rain_drop = db_raw_data_rain_drop.apply(pd.to_numeric)
    db_raw_data_rain_intensity.columns = ['rain_intensity']  
    db_date_time.columns = ['datetime']  
    db_rain = pd.concat([db_raw_data_rain_drop, db_raw_data_rain_intensity, db_date_time], axis=1)
    
    return db_rain

def velo_dia(path):
    file_list=os.listdir(path)        
    pd.options.display.max_colwidth = 12000    
    particle_diameter = np.array([0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187, 1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 
                                  3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500, 9.500, 11.000, 13.000, 15.000, 17.000, 19.000, 21.500, 24.500])
    particle_speed = np.array([0.05, 0.150, 0.250, 0.350, 0.450, 0.550, 0.650, 0.750, 0.850, 0.950, 1.100, 1.300, 1.500, 1.700, 1.900, 2.200, 2.600, 3.000, 
                               3.400, 3.800, 4.400, 5.200, 6.000, 6.800, 7.600, 8.800, 10.400, 12.000, 13.600, 15.200, 17.600, 20.800])
    particle_speed = particle_speed.reshape(32,1)
    particle_speed_t = particle_speed[::-1]
    
    db_raw_data_ave_velo = pd.DataFrame([])    
    db_raw_data_rain_intensity = pd.DataFrame([])
    db_date_time = pd.DataFrame([])
    for i in range(len(file_list)):
        par = pd.read_csv('/'.join([path, file_list[i]]), encoding='euc-kr')
        raw_data = par.loc[80].to_string(index=False)
        raw_data = raw_data.replace(':',';').split(';')
        raw_data = raw_data[1:1025]
        raw_data_re = np.array(raw_data).astype(float)
        raw_data_re = raw_data_re.reshape(32,32)[::-1]
        
        # average wind speed       
        wind_count_sum = np.sum(particle_speed_t*raw_data_re, axis=0)
        sum_count = np.sum(raw_data_re, axis=0)        
        ave_velo = np.nan_to_num(wind_count_sum/sum_count, copy=True).reshape(1,32)
                
        # rain intensity
        raw_data_rain_intensity = float(par.columns.tolist()[0].split(':')[1])
        
        # date
        time = np.array(par.loc[18].to_string(index=False).split(':')).astype(int)
        date = np.array(par.loc[19].to_string(index=False).replace('.',':').split(':')).astype(int)
        date_time = datetime(date[3],date[2],date[1],time[1],time[2],time[3])
        
        db_raw_data_ave_velo = db_raw_data_ave_velo.append(pd.DataFrame(ave_velo), ignore_index = True)
        db_raw_data_rain_intensity = db_raw_data_rain_intensity.append(pd.Series([raw_data_rain_intensity]), ignore_index = True)
        db_date_time = db_date_time.append([date_time], ignore_index = True)
    
    db_raw_data_ave_velo.columns = particle_diameter
    db_raw_data_ave_velo = db_raw_data_ave_velo.apply(pd.to_numeric)
    db_raw_data_rain_intensity.columns = ['rain_intensity']  
    db_date_time.columns = ['datetime']  
    db_wind = pd.concat([db_raw_data_ave_velo, db_raw_data_rain_intensity, db_date_time], axis=1)
    
    return db_wind

def find_mis(path):
    file_list=os.listdir(path)
    file_list_mis=[file for file in file_list if file.endswith(".mis")]
    return file_list_mis