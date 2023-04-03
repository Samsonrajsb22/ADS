#!/usr/bin/env python
# coding: utf-8

# In[138]:


"""
Created on Fri Mar 24 17:45:18 2023

@author: Samson Raj Babu Raj - 22013145
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats import kurtosis


# In[115]:


def clean(a):
    
    """
    The function fills the missing value with 0.

    Parameters:
    a (pandas DataFrame): Dataframe to clean.

    Returns:
    Cleaned data """
    
    # count the number of missing values in each column of the DataFrame
    a.isnull().sum()
    # fill any missing values with 0 and update the DataFrame in place
    a.fillna(0,inplace=True)
    
    return


# In[134]:


clean(Coun)


# In[124]:


def stats(b):
    
    """
    This function performs staistical operation for the cleaned Dataframe - Skew, Kurtosis,describe,Co-variance functions.
    
    """
    
    # extract the columns from the 5th column onward and assign to variable "stats"
    stats=b.iloc[:,5:]
    
    # calculate the skewness,kurtosis and Covariance  
    print(skew(stats, axis=0, bias=True),kurtosis(stats, axis=0, bias=True),stats.describe(),stats.cov())


# In[145]:


stats(Coun)


# In[144]:


Coun.describe()


# In[125]:


with open ("C:/Users/hp/Desktop/ADS1/New folder/API_19_DS2.csv") as g:
    g
g

API_data1 = pd.read_csv("C:/Users/hp/Desktop/ADS1/New folder/API_19_DS2.csv", index_col = 0, encoding = 'cp1252', header = 4)
API_data2 = API_data1.replace(np.NaN, 0)
API_data2

Country_data1 = pd.read_csv("C:/Users/hp/Desktop/ADS1/New folder/Country_API_19_DS2.csv", index_col = 0)
Country_data1.replace(np.NaN, "None")
Country_data = Country_data1.drop("Unnamed: 5",axis=1)
Country_data

Indicator_data1 = pd.read_csv("C:/Users/hp/Desktop/ADS1/New folder/Indicator_API_19_DS2.csv", index_col = 0)
Indicator_data1.replace(np.NaN, 0)
Indicator_data = Indicator_data1.drop("Unnamed: 4", axis=1)
Indicator_data


# In[126]:


Merged_data


# In[177]:


API_Cou = API_data1.merge(Country_data, left_on = "Country Name", right_on = "TableName" , how= "left", suffixes = ("_API","_Country")) 
API_Cou

API_Cou_Indic = API_Cou.merge(Indicator_data, left_on = "Indicator Name" ,right_on = "INDICATOR_NAME", how = "right" ,suffixes=("_API_Cou","_Indic" ))
API_Cou_Indic

Merged_data = API_Cou_Indic[["Country Code", "TableName", "Indicator Name", "Indicator Code","2010","2011","2012","2013","2014","2015"]]
Merged_data

full_data = API_Cou_Indic[["Country Code", "TableName", "Indicator Name", "Indicator Code","2010","2011","2012","2013","2014","2015",'2016','2017','2018','2019','2020','2021']]
full_data

Ind_list = ("EG.ELC.NGAS.ZS" , "EG.ELC.NUCL.ZS")
# "EG.ELC.PETR.ZS","EG.ELC.HYRO.ZS", "EG.ELC.COAL.ZS"
# Reso_list = ("Petroleum", "Nuclear","Natural Gas", "Hydro", "Coal")
# Ind_list = Reso_list

Edu = Merged_data[Merged_data["Indicator Code"].isin(Ind_list)]
Edu.dropna(inplace=True)  # drop rows with missing values
Edu
Coun = ("IND", "BRA", "CAN", "SWE")
Coun = Edu[Edu["Country Code"].isin(Coun)]
Coun
# # Re_data = Coun.set_index(["TableName", "Country Code","Indicator Code", "Indicator Name"])
# # #Re_data = Coun.set_index("TableName")
# # Re_data

grouped = Coun.groupby(['Country Code', "Indicator Name"]).mean()
grouped

grouped.plot(kind='bar')
bar_width = 0.5
# set plot title and axis labels
plt.title('Mean Electricity production - 2010 to 2015 ')
plt.xlabel('Country Code, Indicator Code')
plt.ylabel('Mean Value')

# show the plot
plt.show()


# In[128]:


# group the data by country and sum the values for each resource
grouped = Coun.groupby('Country Code').sum()

# plot a pie chart for each resource in the dataset
for col in grouped.columns:
    plt.figure()
    grouped[col].plot(kind='pie', title='Total Percentage of Electricity produced', autopct='%1.1f%%')
    plt.ylabel('Percentage')
    plt.show()


# In[178]:


Ind_list = ("SP.URB.TOTL.IN.ZS","SP.POP.GROW")
Pop_Edu = full_data[full_data["Indicator Code"].isin(Ind_list)]
Pop_Edu = Pop_Edu.set_index(["Country Code"])
Pop_Edu


# In[ ]:





# In[ ]:





# In[179]:


#2nd plot for the --------

Ind = Pop_Edu[Pop_Edu['TableName'] == 'India']
Ind
Ind_Pop_sum = Ind[Ind['Indicator Code'] == "SP.POP.GROW"]
#AF_pop_sum = pd.DataFrame(AF_pop_sum)
Ind_data = Ind_Pop_sum[['2010','2011','2012', '2013', '2014', '2015','2016','2017','2018','2019','2020','2021']]
Ind_data

Ind_df = Ind_data.T
Ind_df = Ind_df.rename_axis("YEAR")
Ind_df
# IndiaPop = Ind_df.rename(columns = {2662: "India"})
# IndiaPop

# #Brazil


Br = Pop_Edu[Pop_Edu['TableName'] == 'Brazil']
Br
Br_Pop_sum = Br[Br['Indicator Code'] == "SP.POP.GROW"]
#AF_pop_sum = pd.DataFrame(AF_pop_sum)
Br_data = Br_Pop_sum[['2010','2011','2012', '2013', '2014', '2015','2016','2017','2018','2019','2020','2021']]
Br_data

Br_df = Br_data.T
Br_df = Br_df.rename_axis("YEAR")
Br_df
# BrPop = Br_df.rename(columns = {2662: "India"})
# BrPop

# Canada
Ca = Pop_Edu[Pop_Edu['TableName'] == 'Canada']
Ca
Ca_Pop_sum = Ca[Ca['Indicator Code'] == "SP.POP.GROW"]
#AF_pop_sum = pd.DataFrame(AF_pop_sum)
Ca_data = Ca_Pop_sum[['2010','2011','2012', '2013', '2014', '2015','2016','2017','2018','2019','2020','2021']]
Ca_data

Ca_df = Ca_data.T
Ca_df = Ca_df.rename_axis("YEAR")
Ca_df
# CaPop = Ca_df.rename(columns = {2662: "India"})
# CaPop


# #Sweden

Sw = Pop_Edu[Pop_Edu['TableName'] == 'Sweden']
Sw
Sw_Pop_sum = Sw[Sw['Indicator Code'] == "SP.POP.GROW"]
#AF_pop_sum = pd.DataFrame(AF_pop_sum)
Sw_data = Sw_Pop_sum[['2010','2011','2012', '2013', '2014', '2015','2016','2017','2018','2019','2020','2021']]
Sw_data

Sw_df = Sw_data.T
Sw_df = Sw_df.rename_axis("YEAR")
Sw_df
# SwPop = Ca_df.rename(columns = {2662: "India"})
# SwPop


# In[181]:


Merged_data_Edu = Ind_df.merge(Br_df, how="outer", on="YEAR")               .merge(Ca_df, how="outer", on="YEAR")               .merge(Sw_df, how="outer", on="YEAR") 
Merged_data_Edu


# In[192]:


full_data


# In[183]:


# create the bar plot
Merged_data_Edu.plot.bar()

# set the title and axis labels
plt.title('Total Population(2010-2022)')
plt.xlabel('Year')
plt.ylabel('Total Population in Millions')

# display the plot
plt.show()


# In[210]:


def heatmap_india(x):
    
    """
    A function that creates a heatmap of the correlation matrix between different indicators for China.

    Args:
    x (pandas.DataFrame): A DataFrame containing data on different indicators for various countries.

    Returns:
    This function plots the heatmap ."""
    
    # Specify the indicators to be used in the heatmap
    indicator=['Population growth (annual %)',
               'Mortality rate, under-5 (per 1,000 live births)',
               'Population, total',
               'Urban population',
               'Access to electricity (% of population)']
    
#     # Filter the data to keep only China's data and the specified indicators
    Can=x.loc[x['TableName']=='Canada']
    Can = Can[Can['Indicator Name'].isin(indicator)]
#     # Pivot the data to create a DataFrame with each indicator as a column
    Can_df = india.pivot_table(Can,columns= x['Indicator Name'])
#     # Compute the correlation matrix for the DataFrame
    can_df.corr()
    # Plot the heatmap using seaborn
    plt.figure(figsize=(12,8))
    sns.heatmap(Can_df.corr(),fmt='.2g',annot=True,cmap='magma',linecolor='black')
    plt.title('Canada',fontsize=15,fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    plt.show()


# In[212]:


heatmap_Can(full_data)


# In[ ]:




