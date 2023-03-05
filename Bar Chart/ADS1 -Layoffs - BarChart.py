# -*- coding: utf-8 -*-
"""
Spyder Editor

Generated Bar Chart for the Visulisation Assignment
Author: Samson Raj Babu Raj


"""
# importing packages

import pandas as pd
import matplotlib.pyplot as plt

# Reading CSV file and performing data cleaning and sort fucntions
df_layoff = pd.read_csv(
    "C:/Users/hp/Desktop/ADS1/Bar Chart/Tech_Layoff_Data.csv")
df_layoff.head()
df_layoff.isnull()
df_layoff["total_layoffs"] = df_layoff["total_layoffs"].replace(
    to_replace="Unclear", value="0")
df_layoff
df_layoff_sort = df_layoff[["company", "total_layoffs"]]
df_layoffs = df_layoff_sort.sort_values(
    "total_layoffs", ascending=False).head(5)

# plotting bar graph
plt.bar(df_layoffs["company"], df_layoffs["total_layoffs"])
# adding the x label
plt.xlabel("Company")
# adding the y label
plt.ylabel("No.of.Layoffs")
# adding the title
plt.title("Companies Layoffs survey 2022-2023", color="Green")
plt.grid = True
plt.show()
