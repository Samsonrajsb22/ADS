# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:45:18 2023

@author: Samson Raj Babu Raj - 22013145
"""

# Importing packages
import pandas as pd  # import pandas as pd
import numpy as np  # import numpy as np
import matplotlib.pyplot as plt  # import matplot as plt
import seaborn as sns  # import seaborn as sns
from scipy.stats import skew  # import skew from stats as skew
from scipy.stats import kurtosis  # import kurtosis from stats as kurtosis


# Function to clean the data
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
    a.fillna(0, inplace=True)

    return


# function for stats analysis
def stats(b):
    """
    This function takes a pandas DataFrame `a` and performs several statistical calculations on the columns.
    It prints the summary statistics, correlation matrix, skewness, and kurtosis
    for the selected columns.  """

    # extract the columns from the 4th column onward and assign to variable "stats"
    stats = b.iloc[:, 4:]

    # calculate the skewness,kurtosis and Covariance
    print(skew(stats, axis=0, bias=True), kurtosis(
        stats, axis=0, bias=True), stats.describe(), stats.cov())


stats(Merged_data)  # shows the skewness,kurtosis,describe for the dataframe.


# Reading CSV file
with open("C:/Users/hp/Desktop/ADS1/New folder/API_19_DS2.csv") as fp:
    fp
fp

API_data1 = pd.read_csv("C:/Users/hp/Desktop/ADS1/New folder/API_19_DS2.csv",
                        index_col=0, encoding='cp1252', header=4)
API_data2 = API_data1.replace(np.NaN, 0)  # replace NaN with 0
API_data2  # show the dataframe

Country_data1 = pd.read_csv(
    "C:/Users/hp/Desktop/ADS1/New folder/Country_API_19_DS2.csv", index_col=0)
Country_data1.replace(np.NaN, "None")
Country_data = Country_data1.drop("Unnamed: 5", axis=1)  # remove column
Country_data  # show the dataframe

Indicator_data1 = pd.read_csv(
    "C:/Users/hp/Desktop/ADS1/New folder/Indicator_API_19_DS2.csv", index_col=0)
Indicator_data1.replace(np.NaN, 0)  # replace NaN with 0
Indicator_data = Indicator_data1.drop("Unnamed: 4", axis=1)  # remove column
Indicator_data  # show the dataframe


# Merge the dataframe
API_Cou = API_data1.merge(Country_data, left_on="Country Name",
                          right_on="TableName", how="left", suffixes=("_API", "_Country"))
API_Cou  # Merge API and Country Dataframe

API_Cou_Indic = API_Cou.merge(Indicator_data, left_on="Indicator Name",
                              right_on="INDICATOR_NAME", how="right", suffixes=("_API_Cou", "_Indic"))
API_Cou_Indic  # Merge API,Country,Inidcator Dataframe

Merged_data = API_Cou_Indic[["Country Code", "TableName", "Indicator Name",
                             "Indicator Code", "2010", "2011", "2012", "2013", "2014", "2015"]]
Merged_data  # shows the Merged data from 2010-2015

full_data = API_Cou_Indic[["Country Code", "TableName", "Indicator Name", "Indicator Code",
                           "2010", "2011", "2012", "2013", "2014", "2015", '2016', '2017', '2018', '2019', '2020', '2021']]
full_data        # shows the Merged data from 2010-2021


# Creating and filtering the required data from the data frame
Ind_list = ("EG.ELC.COAL.ZS", "EG.ELC.HYRO.ZS")  # create list of indicator
# check the indicator are in the dataframe

Edu = Merged_data[Merged_data["Indicator Code"].isin(Ind_list)]
Edu.dropna(inplace=True)  # drop rows with missing values
Edu  # shows the Edu Dataframe
Coun = ("IND", "BRA", "CAN", "SWE")  # create list of countries
# check the countries are in the dataframe
Coun = Edu[Edu["Country Code"].isin(Coun)]
Coun  # shows the Coun Dataframe

# groupby function
grouped = Coun.groupby(['Country Code', "Indicator Name"]).mean()

# plot bar graph for filtered data
plt.figure(dpi=300)
grouped.plot(kind='bar')  # kind of the graph
bar_width = 2  # width of the bar
# title for the bar graph
plt.title('Mean Electricity production - 2010 to 2015')
plt.xlabel('Country Code, Indicator Code')  # xlabel name
plt.ylabel('Mean Value')  # ylabel name
plt.show()  # show the bar graph


# group the data by country
grouped = Coun.groupby('Country Code').sum()


# plot a pie chart for each resource in the dataset
for col in grouped.columns:
    plt.figure(dpi=300)
    grouped[col].plot(
        kind='pie', title='Total Percentage of Electricity produced', autopct='%1.1f%%')
    plt.ylabel('Percentage')   # ylabel name
    plt.show()

Ind_list = ("SP.URB.TOTL.IN.ZS", "SP.POP.GROW")  # list for the indicator code
Pop_Edu = full_data[full_data["Indicator Code"].isin(
    Ind_list)]  # check the filter in dataframe
Pop_Edu = Pop_Edu.set_index(["Country Code"])  # set index
Pop_Edu  # shows the filtered dataframe


# filter from dataframe - India
Ind = Pop_Edu[Pop_Edu['TableName'] == 'India']
Ind
Ind_Pop_sum = Ind[Ind['Indicator Code'] == "SP.POP.GROW"]
Ind_data = Ind_Pop_sum[['2010', '2011', '2012', '2013', '2014',
                        '2015', '2016', '2017', '2018', '2019', '2020', '2021']]
Ind_data
# Transposing dataframe
Ind_df = Ind_data.T
Ind_df = Ind_df.rename_axis("YEAR")
Ind_df

# filter from dataframe - Brazil
Br = Pop_Edu[Pop_Edu['TableName'] == 'Brazil']
Br
Br_Pop_sum = Br[Br['Indicator Code'] == "SP.POP.GROW"]
Br_data = Br_Pop_sum[['2010', '2011', '2012', '2013', '2014',
                      '2015', '2016', '2017', '2018', '2019', '2020', '2021']]
Br_data
# Transposing dataframe
Br_df = Br_data.T
Br_df = Br_df.rename_axis("YEAR")
Br_df

# #filter from dataframe - Canada
Ca = Pop_Edu[Pop_Edu['TableName'] == 'Canada']
Ca
Ca_Pop_sum = Ca[Ca['Indicator Code'] == "SP.POP.GROW"]
Ca_data = Ca_Pop_sum[['2010', '2011', '2012', '2013', '2014',
                      '2015', '2016', '2017', '2018', '2019', '2020', '2021']]
Ca_data
# Transposing dataframe
Ca_df = Ca_data.T
Ca_df = Ca_df.rename_axis("YEAR")
Ca_df


# #filter from dataframe - Sweden
Sw = Pop_Edu[Pop_Edu['TableName'] == 'Sweden']
Sw
Sw_Pop_sum = Sw[Sw['Indicator Code'] == "SP.POP.GROW"]
Sw_data = Sw_Pop_sum[['2010', '2011', '2012', '2013', '2014',
                      '2015', '2016', '2017', '2018', '2019', '2020', '2021']]
Sw_data
# Transposing dataframe
Sw_df = Sw_data.T
Sw_df = Sw_df.rename_axis("YEAR")
Sw_df

# Merge the Countries and the data in a single dataframe

Merged_data_Edu = Ind_df.merge(Br_df, how="outer", on="YEAR") \
    .merge(Ca_df, how="outer", on="YEAR") \
    .merge(Sw_df, how="outer", on="YEAR") \

# create the bar plot
plt.figure(dpi=300)
Merged_data_Edu.plot.bar()
# set the title and axis labels
plt.title('Total Population(2010-2022)')
plt.xlabel('Year')
plt.ylabel('Total Population in Millions')
plt.show()   # display the plot

# Specify the indicators for the plot and analytics
indicators = ['Population growth (annual %)',
              'Mortality rate, under-5 (per 1,000 live births)',
              'Population, total',
              'Urban population',
              'Access to electricity (% of population)']

# Filter the data to keep only India's data and the specified indicators
India = full_data.loc[full_data['TableName'] == 'India']
India = India[India['Indicator Name'].isin(indicators)]
India = India.T
pd.DataFrame(India)
India
In = India.drop(['Country Code', 'TableName', 'Indicator Name'], axis=0)
pd.DataFrame(In)

Ind = In.rename(columns={375: 'SP.URB.TOTL', 907: 'SP.POP.TOTL',
                1173: 'SP.POP.GROW', 2237: 'SH.DYN.MORT', 15010: 'EG.ELC.ACCS.ZS'})
Ind.drop(["Indicator Code"], inplace=True)
pd.DataFrame(Ind)

# performing maths fucntions
Ind.fillna(0)
Ind['SP.POP.TOTL'] = Ind['SP.POP.TOTL'] / 1000000
Ind['SP.URB.TOTL'] = Ind['SP.URB.TOTL'] / 1000000
Ind['SP.POP.GROW'] = Ind['SP.POP.GROW'] * 100

# scatter plot to find the strength between population growth and electricty access
sns.scatterplot(x="SP.POP.TOTL", y="EG.ELC.ACCS.ZS", data=Ind)
plt.xlabel("India's Total Population(in Millions)")
plt.ylabel("Access to electricity (Total % of population)")
plt.title("Relationship between Total Population vs Access to Electricity in India")
plt.show()

# Statistical operation - standard Deviation for Total population
Std_Pop = np.std(Ind['SP.POP.TOTL'], ddof=1)

# Statistical operation - Variance for Total Population
Var_Pop = np.var(Ind['SP.POP.TOTL'], ddof=1)

# Statistical operation - Standard Deviation for Electricity access
Std_Ele = np.std(Ind['EG.ELC.ACCS.ZS'], ddof=1)

# Statistical operation - Variance for Electricity access
Var_Ele = np.var(Ind['EG.ELC.ACCS.ZS'], ddof=1)

# quantile for Population Total
qua_Pop = np.quantile(Ind['SP.POP.TOTL'], np.linspace(0, 0.50, 4))

# quantile for Electricity access
qua_Ele = np.quantile(Ind['EG.ELC.ACCS.ZS'], np.linspace(0, 0.50, 4))

# Boxplot for Total Population in India for the last decade
plt.figure(dpi=300)  # set dpi
fig, ab = plt.subplots()
ab.boxplot(Ind['SP.POP.TOTL'])  # boxplot for the indicator
ab.set_xticklabels([1])  # set x ticks to 1
# title for the box plot
ab.set_title('Box Plot-Total Population in India for the last decade')
ab.set_ylabel('India Total Population in Millions')    # y label
plt.figure(figsize=(2, 2))
plt.show()  # show the figure

# Boxplot for Electricity access
plt.figure(dpi=300)
fig, yy = plt.subplots()
yy.boxplot(Ind['EG.ELC.ACCS.ZS'])  # boxplot for the indicator
yy.set_xticklabels([1])
yy.set_ylabel('Access to Electricity')    # set y label
# title for the box plot
yy.set_title('Box Plot-Electricity Access percentage in last decade')
yy.axhline(y=qua_Ele[0], color='r', linestyle='-')   # linestyle 1
yy.axhline(y=qua_Ele[1], color='y', linestyle='--')  # linestyle 2
yy.axhline(y=qua_Ele[2], color='b', linestyle='--')  # linestyle 3
yy.axhline(y=qua_Ele[3], color='g', linestyle='-')   # linestyle 4
plt.figure(figsize=(2, 2))
plt.show()  # show the figure
