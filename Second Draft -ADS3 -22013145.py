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

#mean and standard deviation for each indicator
means = data.mean()  #mean of the dataframe
stds = data.std()    #standart deviation of the dataframe
# bar plot
fig, ax = plt.subplots()
ax.bar(range(len(means)), means, yerr=stds, tick_label=means.index) #set the range to plot
ax.set_ylabel("Normalized values")  # set the y label
ax.set_title("Mean and standard deviation of indicators") #set the title

#Scatter plot of GDP per capita and energy use per capita
fig, ax = plt.subplots()
ax.scatter(data_norm["GDP_per_capita"], data_norm["Energy_use_per_capita"])  #set the data range
ax.set_xlabel("GDP per capita (PPP, inflation-adjusted)")  #set the x label
ax.set_ylabel("Energy use per capita") # set the y label
ax.set_title("Scatter plot of GDP per capita and energy use per capita") #set the title

# Set the date range
dates = (pd.to_datetime('2015'), pd.to_datetime('2019'))
# Get CO2 emissions data for selected countries
data_countries = wbdata.get_dataframe({'EN.ATM.CO2E.KT': 'CO2_emissions'},country=['USA', 'CAN', 'MEX', 'GBR', 'IND'],
                                       data_date=dates)
# Remove the missing data
data_countries = data_countries.dropna()
# Pivot the data to get CO2 emissions per country per year
co2_data = data_countries.pivot_table(index='date', columns='country', values='CO2_emissions')

# Increase the size of the figure
plt.figure(figsize=(100, 12))
# Plot the data as a clustered bar chart
co2_data.plot(kind='bar', alpha=0.9)
plt.xlabel('Year') # set the x label
plt.ylabel('CO2 Emissions (kt)') # set the y label 
plt.title('CO2 Emissions by Country and Year')  #vset the Title
plt.show() # show the plot

# Set the date range for the data
date_range = (pd.to_datetime('2015'), pd.to_datetime('2019'))
# Get GDP per capita data for selected countries from the World Bank
data_countries = wbdata.get_dataframe({"NY.GDP.PCAP.PP.KD": "GDP_per_capita"},
                                       country=['USA', 'CAN', 'MEX', 'GBR', 'IND'],
                                       data_date=date_range)
# Remove the missing data
data_countries = data_countries.dropna()
# Pivot the data to get GDP per capita per country per year
gdp_data = data_countries.pivot_table(index='date', columns='country', values='GDP_per_capita')
# Increase the size of the plot
plt.figure(figsize=(100, 12))
# Plot the data as a clustered bar chart
gdp_data.plot(kind='bar', alpha=0.9)
# Add labels and title to the plot
plt.xlabel('Year') # set the x label
plt.ylabel('GDP per capita (constant 2017 international $)') # set the y label
plt.title('GDP per Capita by Country and Year') # set the title
plt.show() # Show the plot

# Set the date range
dates = (pd.to_datetime('2011'), pd.to_datetime('2015'))
# Get energy use per capita data for selected countries
data_countries = wbdata.get_dataframe({"EG.USE.PCAP.KG.OE": "Energy_use_per_capita"},country=['USA', 'CAN', 'MEX', 'GBR', 'IND'], data_date=dates)
# Remove the missing data
data_countries = data_countries.dropna()
# Pivot the data to get energy use per capita per country per year
energy_data = data_countries.pivot_table(index='date', columns='country', values='Energy_use_per_capita')
# Increase the size of the figure
plt.figure(figsize=(100, 12))
# Plot the data as a clustered bar chart
energy_data.plot(kind='bar', alpha=0.9)
plt.xlabel('Year') # set the x label
plt.ylabel('Energy Use per Capita (kg of oil equivalent)') # set the y label
plt.title('Energy Used per Capita by Country and Year') # set the title
plt.show()  # show the plot

#Histogram of CO2 emissions per capita
fig, ax = plt.subplots()
ax.hist(data_norm["CO2_emissions_per_capita"], bins=20) # plot the histogram
ax.set_xlabel("CO2 emissions per capita")  #set the x label
ax.set_ylabel("Frequency") # set the y label
ax.set_title("Histogram of CO2 emissions per capita") # set the Title

#Scatter plot matrix of all indicators
pd.plotting.scatter_matrix(data, diagonal="hist")
plt.suptitle("Scatter plot matrix of indicators")

#Heatmap of correlations between indicators
fig, ax = plt.subplots()
corr = data.corr() # correlation
im = ax.imshow(corr, cmap="coolwarm")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right") # set the x label
ax.set_yticklabels(corr.index) # set the y label
plt.colorbar(im, ax=ax) # set the color
ax.set_title("Correlation between indicators") # set the title

# Calculate the percentage of countries in each cluster
cluster_counts = data["Cluster"].value_counts(normalize=True) * 100
# Plot a pie chart of the cluster percentages
fig, ax = plt.subplots()
ax.pie(cluster_counts, labels=["Low GDP, Low CO2 emissions", "Medium GDP, Medium CO2 emissions", "High GDP, High CO2 emissions"], autopct="%1.1f%%")
ax.set_title("Percentage of Countries by GDP and CO2 Emissions Cluster") # set the title
plt.show() # show the plot

data_countries = wbdata.get_dataframe({"EN.ATM.CO2E.KT": "CO2_emissions"}, country=["USA", "CAN", "MEX","GBR","IND"])

# Remove rows with missing data
data_countries = data_countries.dropna()

# Pivot the data to get CO2 emissions per country per year
co2_data = data_countries.pivot_table(index="country", columns="date", values="CO2_emissions")
data_2019 = co2_data.filter(like="2019")
print(data_2019.head())

# Plot the data as a bar chart
data_2019.plot(kind="bar", legend=None)
plt.xlabel("Country") # set the x label
plt.ylabel("CO2 Emissions per Capita (metric tons)") # set the y label
plt.title("CO2 Emissions per Capita by Country in 2019") # set the title
plt.show() #show the plot