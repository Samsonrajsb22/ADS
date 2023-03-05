
"""
Spyder Editor

Generated Line plot for the Visulisation Assignment
Author: Samson Raj Babu Raj

"""
# importing packages

import pandas as pd
import matplotlib.pyplot as plt

# Reading the CSV file

df_canada = pd.read_csv('C:/Users/hp/Desktop/ADS1/Line plot/canadian_immigration_data.csv',
                        index_col="Country")
print(df_canada.head())

# adjusting the figure size
plt.figure(figsize=(10, 5))

# creating a list of all the years from 1980 to 2014 by calling the
# list function and using map command to iterate the values then
# specifying the type of data and then calling its range
years = list(map(str, range(2005, 2014)))

# plotting the graph
df_canada.loc["India", years].plot(kind="line")
df_canada.loc["Pakistan", years].plot(kind="line")
df_canada.loc["Bangladesh", years].plot(kind="line")
df_canada.loc["Sri Lanka", years].plot(kind="line")
plt.legend()

# adding the title
plt.title("Immigration from Asian Countries")

# adding the x label
plt.xlabel("Years")

# adding the y label
plt.ylabel("Number of Immigrants")
plt.show()
