# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:48:34 2020

@author: user
"""

import parsivel_processer as parp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import os

particle_diameter = np.array([0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187, 1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 
                                  3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500, 9.500, 11.000, 13.000, 15.000, 17.000, 19.000, 21.500, 24.500])
particle_speed = np.array([0.05, 0.150, 0.250, 0.350, 0.450, 0.550, 0.650, 0.750, 0.850, 0.950, 1.100, 1.300, 1.500, 1.700, 1.900, 2.200, 2.600, 3.000, 
                               3.400, 3.800, 4.400, 5.200, 6.000, 6.800, 7.600, 8.800, 10.400, 12.000, 13.600, 15.200, 17.600, 20.800])
def velo_dia_grap(path):
    db_1 = parp.velo_dia(path).dropna(axis=0)
    
    # Data classification (classified into 10 units)
    db_intensity_dia = pd.DataFrame([])
    for i in range(int(round(db_1["rain_intensity"].max(), -1)/10)+1) : 
        db_1_re = db_1.drop(db_1[(db_1["rain_intensity"] < 1)].index)
        del db_1_re["datetime"]
        
        intensity_raw= db_1_re.loc[(db_1_re["rain_intensity"]>10*i)&(db_1_re["rain_intensity"]<10*(i+1)),:]
        average_intensity_raw = pd.DataFrame((intensity_raw).mean()).T    
        data_quan = pd.DataFrame([len(intensity_raw)], columns=['data_quan'])
        db_rain = pd.concat([average_intensity_raw, data_quan], axis=1) 
        db_rain.index=[str(i*10) + ' ~ ' + str(i*10+10)+' mm/h']
        db_intensity_dia = db_intensity_dia.append(db_rain)
    
    db_intensity_dia_re = db_intensity_dia.dropna(axis=0).drop(['data_quan'], axis=1)
    
    file_list=os.listdir(path)
    density_function_df = pd.DataFrame([])
    sum_vol_density_df = pd.DataFrame([])
    raw_data_re_df = []
    raw_data_sum_df = pd.DataFrame([])
        
    for i in range(len(file_list)):
        density_function_array, sum_vol_speed_density_array, raw_data_re_array = parp.density_function(path, file_list[i])
        density_function = pd.DataFrame([density_function_array])
        sum_vol_speed_density = pd.DataFrame([sum_vol_speed_density_array])
        raw_data_re = np.array(raw_data_re_array)
        raw_data_sum = pd.DataFrame(pd.DataFrame(raw_data_re).sum()).T    
        
        density_function_df = density_function_df.append(density_function, ignore_index = True)
        sum_vol_density_df = sum_vol_density_df.append(sum_vol_speed_density, ignore_index = True)
        raw_data_re_df.append(raw_data_re)
        raw_data_sum_df = raw_data_sum_df.append(raw_data_sum, ignore_index = True)
        
    raw_data_re_df = np.array(raw_data_re_df)
    density_function_df.columns = particle_diameter
    sum_vol_density_df.columns = particle_diameter
    raw_data_sum_df.columns = particle_diameter
    
    db_rain = parp.pars_db(path).dropna(axis=0)
    raw_data_sum_df = pd.concat([raw_data_sum_df, db_rain["rain_intensity"], db_rain["datetime"]], axis=1)
    
    # Data classification (classified into 10 units)
    db_raw_data_sum_df = pd.DataFrame([])
    for i in range(int(round(raw_data_sum_df["rain_intensity"].max(), -1)/10)+1) : 
        raw_data_sum_df_re = raw_data_sum_df.drop(raw_data_sum_df[(raw_data_sum_df["rain_intensity"] < 1)].index)
        del raw_data_sum_df_re["datetime"]
        
        intensity_raw= raw_data_sum_df_re.loc[(raw_data_sum_df_re["rain_intensity"]>10*i)&(raw_data_sum_df_re["rain_intensity"]<10*(i+1)),:]
        average_intensity_raw = pd.DataFrame(intensity_raw.mean()).T
        data_quan = pd.DataFrame([len(intensity_raw)], columns=['data_quan'])
        db_rain = pd.concat([average_intensity_raw, data_quan], axis=1) 
        db_rain.index=[str(i*10) + ' ~ ' + str(i*10+10)+' mm/h']
        db_raw_data_sum_df = db_raw_data_sum_df.append(db_rain)
    
    db_raw_data_sum_df = db_raw_data_sum_df.dropna(axis=0).drop(['data_quan'], axis=1)    
    db_raw_data_sum_dfs = db_raw_data_sum_df.sum()
    db_raw_data_sum_dfs["rain_intensity"] = np.average(db_raw_data_sum_df["rain_intensity"])
    
    db_intensity_dia_res = db_intensity_dia_re.sum()/len(db_intensity_dia_re)
    db_intensity_dia_res["rain_intensity"] = np.average(db_intensity_dia_res["rain_intensity"])
    
    # Calculate mean and median velocity
    db_raw_data_sum_dfs = db_raw_data_sum_dfs.drop(["rain_intensity"])
    db_intensity_dia_res = db_intensity_dia_res.drop(["rain_intensity"])
    
    for i in range(1):
        temp = round(db_raw_data_sum_dfs)
           
        temps = []
            
        for idx, val in enumerate(temp.index):        
            temp_alts = []
            for a in range(int(temp[val])):
                temp_alts.append(db_intensity_dia_res[val])
            temps.append(temp_alts)
        temps = pd.DataFrame(np.array(pd.DataFrame(temps)).flatten()).dropna()
    
    print("Mean veolcity  : " + str(round(np.sum(db_raw_data_sum_dfs*db_intensity_dia_res)/np.sum(db_raw_data_sum_dfs),2)) + " m/s")
    print("Median veolcity : " + str(round(np.median(temps),3)) + " m/s")
    
    return db_intensity_dia_res

# Regression equations (Edward, 2002)
velo_reg_eq = (-0.1021 + 4.932*particle_diameter - 0.9551*(particle_diameter**2) + 0.07934*(particle_diameter**3)-0.00236*(particle_diameter**4)).reshape(1,32)
velo_reg_eq = pd.DataFrame(velo_reg_eq, columns=particle_diameter, index=['Edward, 2002'])

# Graph
## Graph settings
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

## Graph (average velocity vs particle diameter)
plt.figure(num=None,  dpi=500, facecolor='w', edgecolor='k') 
plt.plot(particle_diameter, velo_dia_grap('C:\\Users\\10mm\\Area A'), label = "10 mm/h")  ## YOU MUST CHANGE DATA PATH ##
plt.plot(particle_diameter, velo_dia_grap('C:\\Users\\20mm\\Area A'), label = "20 mm/h")  ## YOU MUST CHANGE DATA PATH ##
plt.plot(particle_diameter, velo_dia_grap('C:\\Users\\30mm\\Area A'), label = "30 mm/h")  ## YOU MUST CHANGE DATA PATH ##
plt.plot(particle_diameter, velo_dia_grap('C:\\Users\\40mm\\Area A'), label = "40 mm/h")  ## YOU MUST CHANGE DATA PATH ##
plt.plot(particle_diameter, velo_dia_grap('C:\\Users\\50mm\\Area A'), label = "50 mm/h")  ## YOU MUST CHANGE DATA PATH ##
    
plt.plot(particle_diameter, velo_reg_eq.T, label = "Edward, 2002", linestyle = 'dashed', color = 'black')
plt.ylabel('Velocity (m/s)')
plt.xlabel('Diameter (mm)')
plt.xlim(0,1.6)
plt.ylim(0,7)
plt.legend(loc='best')
plt.grid()
fig4=plt.gcf()
plt.show()
plt.draw()
fig4.savefig('Velocity_vs_Diameters.png')  
