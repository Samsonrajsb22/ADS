# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:20:52 2023

@author: Name: Samson Raj Babu Raj 
         Student Id : 22013145
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import wbdata
from scipy.stats import t

# Define - Indicators
indicators = {"NY.GDP.PCAP.PP.KD": "GDP_per_capita",
              "EN.ATM.CO2E.PC": "CO2_emissions_per_capita",
              "EG.USE.PCAP.KG.OE": "Energy_use_per_capita"}
# Data and time format
dates = (pd.to_datetime('2011'), pd.to_datetime('2019'))
# Download dataframe
data = wbdata.get_dataframe(indicators, country="all", data_date=dates)
# Remove rows with missing data
data = data.dropna()
# Data Normalization
data_norm = (data - data.mean()) / data.std()
# k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=0).fit(data_norm)
# Add cluster labels as a new column to the dataframe
data["Cluster"] = kmeans.labels_

# Plot the data using different colors for each cluster
fig, bd = plt.subplots()
for cluster, group in data.groupby("Cluster"):
    bd.scatter(group["GDP_per_capita"], group["CO2_emissions_per_capita"],
               label="Cluster {}".format(cluster))
bd.set_xlabel("GDP per capita (PPP, inflation-adjusted)")
bd.set_ylabel("CO2 emissions per capita")
bd.set_title(
    "Country Clusters based on GDP per Capita and CO2 Emissions per Capita")
bd.legend()

# Define the function to fit


def func(x, a, b, c):
    '''The model simplifies by using a low order polynomial function'''
    return a + b*x + c*x**2


# Get data for a specific country (USA) and indicator (CO2 emissions)
data_country = wbdata.get_dataframe(
    {"EN.ATM.CO2E.KT": "CO2_emissions"}, country="USA")
# Drop the missing data
data_country = data_country.dropna()
# Transpose the Dataframe
data_country.T
# Extract the year and CO2 emissions values
x = data_country.index.values.astype(int)  # cast the index to integer type
y = data_country["CO2_emissions"].values
# Fit the function to the data
# increase the number of function evaluations
popt, pcov = curve_fit(func, x, y, maxfev=100000000)

# Compute confidence interval of the fit


def err_ranges(x, y, popt, pcov, alpha=0.05):
    n = len(y)
    p = len(popt)
    dof = max(0, n - p)  # degrees of freedom
    tval = t.ppf(1-alpha/2., dof)  # t-value

    # Residual standard deviation
    residuals = y - func(x, *popt)
    s_err = np.sqrt(np.sum(residuals**2) / dof)

    # Confidence interval
    x_new = np.repeat(x, len(y)).reshape(len(x), len(y)).T
    conf = tval * s_err * \
        np.sqrt(1/n + (x_new - np.mean(x_new, axis=1, keepdims=True))
                ** 2 / np.sum((x - np.mean(x))**2))
    y_pred = func(x, *popt)
    return y_pred - conf, y_pred + conf


y_fit = func(x, *popt)
y_lower, y_upper = err_ranges(x, y, popt, pcov)

# plot the fit curve and 95% confidence interval
y_fit = func(x, *popt)
sigma = np.sqrt(np.diag(pcov))
y_lower = func(x, *(popt - 1.96 * sigma))
y_upper = func(x, *(popt + 1.96 * sigma))

# Plot the data, fit curve, and confidence interval
fig, cs = plt.subplots()   # create sublot
cs.scatter(x, y, label="Data")  # data to plot as scattered plot
cs.plot(x, y_fit, label="Fit")  # data to plot to fit
cs.fill_between(x, y_lower.flatten(), y_upper.flatten(), alpha=0.3,
                label="95% Confidence Interval")  # fill the values with
cs.set_xlabel("Year")  # set the x label
cs.set_ylabel("fit range")  # set the y label
cs.legend()
plt.show()  # show the plot
