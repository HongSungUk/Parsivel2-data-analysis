# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:41:23 2021

@author: Hong Sung Uk (hsu12375@gmail.com / 010-9191-5398)
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
from datetime import datetime
from sklearn.linear_model import LinearRegression
import glob

# Data preprocessing functions
## Rain diameter 32 class (mm) / Rain speed 32 class (m/s)
particle_diameter = [0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187, 1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 
                                  3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500, 9.500, 11.000, 13.000, 15.000, 17.000, 19.000, 21.500, 24.500]
particle_speed = np.array([0.05, 0.150, 0.250, 0.350, 0.450, 0.550, 0.650, 0.750, 0.850, 0.950, 1.100, 1.300, 1.500, 1.700, 1.900, 2.200, 2.600, 3.000, 
                               3.400, 3.800, 4.400, 5.200, 6.000, 6.800, 7.600, 8.800, 10.400, 12.000, 13.600, 15.200, 17.600, 20.800])

## Restores raw data of speed 32 x diameter 32 from msi file
def rebuild_raw_data(file_path):
    pd.options.display.max_colwidth = 12000
    par = pd.read_csv(file_path, encoding='euc-kr')
    raw_data = par.loc[80].to_string(index=False)
    raw_data = raw_data.replace(':',';').split(';')
    raw_data = raw_data[1:1025]
    raw_data_re = np.array(raw_data).astype(float)
    raw_data_re = raw_data_re.reshape(32,32)[::-1]        
    return raw_data_re

## If you enter the path where the msi files are stored in the path, it automatically separates the rainfall amount, date and time, and particle size and organizes them into one table.
def pars_db(file_path): 
    pd.options.display.max_colwidth = 12000
    particle_diameter = [0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187, 1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 
                                  3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500, 9.500, 11.000, 13.000, 15.000, 17.000, 19.000, 21.500, 24.500]
    
    db_raw_data_rain_drop = pd.DataFrame([])    
    db_raw_data_rain_intensity = pd.DataFrame([])
    db_date_time = pd.DataFrame([])
    
    par = pd.read_csv(file_path, encoding='euc-kr')
    
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

# Data input
path = 'C:\\Users\\Desktop\\10mm\\Area A' ## YOU MUST CHANGE DATA PATH ##
file_list = glob.glob(path + '\\**', recursive=False)

## Integrate raw data for each file
raw_data_re_df = []
raw_data_sum_df = pd.DataFrame([])
db_rain = pd.DataFrame([])
for i in range(len(file_list)):
    raw_data_re_array = rebuild_raw_data(file_list[i])
    raw_data_re = np.array(raw_data_re_array)
    raw_data_sum = pd.DataFrame(pd.DataFrame(raw_data_re).sum()).T        
    raw_data_re_df.append(raw_data_re)
    raw_data_sum_df = raw_data_sum_df.append(raw_data_sum, ignore_index = True)
    db_rain_raw = pars_db(file_list[i]).dropna(axis=0)
    db_rain = db_rain.append(db_rain_raw)    
    
raw_data_re_df = np.array(raw_data_re_df)
raw_data_sum_df.columns = particle_diameter

db_rain = db_rain.reset_index(drop=True)
raw_data_sum_df = pd.concat([raw_data_sum_df, db_rain["rain_intensity"], db_rain["datetime"]], axis=1)

## Data classification (in 10 mm/h increments)
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

# Graph
## Graph settings
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

## Graph (Diameter vs Average count of raindrop)
plt.figure(num=None,  dpi=200, facecolor='w', edgecolor='k')
for i in range(len(db_raw_data_sum_df)):
    plt.plot(np.array(particle_diameter), db_raw_data_sum_df.iloc[i].drop(["rain_intensity"]), label = db_raw_data_sum_df.index[i])
    plt.ylabel('Average count of raindrop')
    plt.xlabel('Diameter (mm)')
plt.xlim(0,9)
plt.legend(loc='best')
plt.grid()
fig5=plt.gcf()
plt.show()
plt.draw()

## Cumulative Volume Ratio VMD by Rainfall
particle_volume = 4/3*np.pi*np.array(particle_diameter)**3 # Assume a single rainfall particle to be spherical and calculate its volume

def min_diff_pos_sorted(sorted_array, target):
    idx = np.searchsorted(sorted_array, target)
    idx1 = max(0, idx-1)
    return np.abs(np.array(sorted_array[idx1:idx+1])-target).argmin() + idx1

vol_per = pd.DataFrame([])
vol_col_per =  pd.DataFrame([])

for i in range(len(raw_data_sum_df_re)):
    temp = round(raw_data_sum_df_re.iloc[i].drop(["rain_intensity"]))
    temp_vol = particle_volume*temp
    temp_vol_per = temp_vol/temp_vol.sum()*100
    temp_vol_per = pd.DataFrame([temp_vol_per])
    vol_per = vol_per.append(temp_vol_per)
    temp_vol_per_acc = pd.DataFrame(columns = [raw_data_sum_df_re.index[i]],index = particle_diameter)
    temp_vol_per_acc.iloc[0] = 0
    temp_vol_per_acc.iloc[1] = 0
    for i in range(2,32):
        temp_vol_per_acc.iloc[i] = temp_vol_per_acc.iloc[i-1]+temp_vol_per.T.iloc[i]
    stat_raw = temp_vol_per_acc = pd.DataFrame(temp_vol_per_acc, index = particle_diameter)
    vol_col_per = pd.concat([vol_col_per, stat_raw], axis=1)

D50_db = pd.DataFrame([])
for idx, val in enumerate(vol_col_per.columns):
    vol_col_50_down = vol_col_per[val].iloc[min_diff_pos_sorted(np.array(vol_col_per[val]), 50)-1]
    vol_col_50 = vol_col_per[val].iloc[min_diff_pos_sorted(np.array(vol_col_per[val]), 50)]
    vol_col_50_up = vol_col_per[val].iloc[min_diff_pos_sorted(np.array(vol_col_per[val]), 50)+1]
    
    dia_col_50_down = particle_diameter[min_diff_pos_sorted(np.array(vol_col_per[val]), 50)-1]
    dia_col_50 = particle_diameter[min_diff_pos_sorted(np.array(vol_col_per[val]), 50)]
    dia_col_50_up = particle_diameter[min_diff_pos_sorted(np.array(vol_col_per[val]), 50)+1]
    
    temp_array = np.array([[vol_col_50_down,vol_col_50,vol_col_50_up], [dia_col_50_down,dia_col_50,dia_col_50_up]])
    line_fitter = LinearRegression()
    line_fitter.fit(temp_array[0].reshape(-1,1), temp_array[1])
    D50 = line_fitter.predict([[50]])
    D50_db = D50_db.append([D50])
    print(str(val) +" D50 : " + str(round(D50[0],3)) +" mm")

D50_db.columns = ['D50']
D50_db = D50_db.reset_index(drop=True)
    
rain_data_result = pd.concat([pd.DataFrame(file_list, columns=['file path']),raw_data_sum_df, D50_db], axis=1)

CU_db = pd.DataFrame([])
for i in range(len(rain_data_result)):
    CU = 100*(1 - abs(rain_data_result['rain_intensity'][i] - np.average(rain_data_result['rain_intensity']))/np.average(rain_data_result['rain_intensity']))
    CU_db = CU_db.append([CU])

print("Average D50 : " + str(round(np.average(D50_db),3)) +" mm")
print("Average intensity : " + str(round(np.average(rain_data_result['rain_intensity']),3)) +" mm/h")   
print("Christiansen's Coefficient of Uniformity(CU) : " + str(round(np.average(CU_db),2)))
