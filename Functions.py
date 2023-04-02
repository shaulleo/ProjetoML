#Basic Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import squarify

#To find addresses
import reverse_geocoder as rg
from geopy.geocoders import Nominatim

#Plot Maps
import folium
from folium.plugins import HeatMap, MarkerCluster 

import warnings
warnings.filterwarnings("ignore")

#Functions


# ---------- PRE PROCESSAMENTO

#Extract education level from customer name 
def extract_education(observation):
    name_list = observation.split(' ')
    if len(name_list) > 2:
        if name_list[0] == 'Msc.':
            education = 2
        elif name_list[0] == 'Bsc.':
            education = 1
        else:
            education = 3
    else:
        education = 0
    return education

#Clean the customers name
def clean_names(observation):
    name_list = observation.split(' ')
    if len(name_list) > 2:
        name = str(name_list[1] +' '+ name_list[2])
    else:
        name = str(name_list[0] + ' '+ name_list[1])
    return name

#Separate birthday date into three different columns
def process_birthdate(df, birthdate):
    df[birthdate] = pd.to_datetime(df[birthdate])
    #birthday
    df['birthday'] = df[birthdate].dt.day
    #birthmonth
    df['birthmonth'] = df[birthdate].dt.month
    #birthyear
    year_ = date.today().year 
    df['age'] = int(year_) - df[birthdate].dt.year
    return df

#Retrieve Address based on latitude and longitude of the observation
def get_address(row): 
    geolocator = Nominatim(user_agent='my_app') 
    full_address = geolocator.reverse(f"{row['latitude']}, {row['longitude']}").address 
    return full_address

#Obtain the region based on address
def clean_address(row):
    full_address = row.split(',')
    if len(full_address) >= 4:
        if full_address[-3] == ' Lisboa' or full_address[-3] == 'Lisboa':
            address = full_address[-4]
        else:
            address = full_address[-3]
    else:
        address = full_address[-3]
    if address[0] == ' ':
        address = address[1:]
    return address

#Encode the address by extracting the mean value of lat/long for every region
def encode_address(dataframe, latitude, longitude, address):
    lat_map = dataframe.groupby(address)[latitude].mean().to_dict()
    long_map = dataframe.groupby(address)[longitude].mean().to_dict()
    dataframe['latitude_encoded'] = dataframe[address].map(lat_map)
    dataframe['longitude_encoded'] = dataframe[address].map(long_map)
    return dataframe

#Encode binary values based on a condition
#Se a condição for true, a col_name tem valor 0, caso contrário tem 1
def binary_encoding(df, col_name, condition):
    #new_col_name = col_name + '_encoded'
    df[col_name] = np.where(condition, 0, 1)
    return df

#Convert several columns into integers
def integer_convert(df, cols):
    df[cols] = df[cols].astype('int64')
    return df


#------- VISUALIZAÇÃO

#Plot Histograms based on the optimum number of bins
def plot_histograms(df, cols):
    for col in cols:
        data = df[col]
        if data.dtype == 'int64':
            bins = data.nunique()
        elif data.dtype == 'float64':
            q75, q25 = np.percentile(data, [75 ,25])
            iqr = q75 - q25
            bin_width = 2 * iqr * len(data)**(-1/3)
            bins = int((data.max() - data.min()) / bin_width)
        else:
            print(f"Skipping column '{col}' - not a numeric datatype")
            continue
        
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.histplot(x=data, bins=bins, color='lightblue', ax=ax)
        ax.set_title(f'{col} histogram', fontsize=14)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.show()

#Plot Bar charts 
def plot_bar_charts(df, columns):
    for col in columns:
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.countplot(x=col, data=df, ax=ax, color='lightblue', linewidth=1, edgecolor=".2")
        ax.set_title(f'{col}', fontsize=12)
        ax.set_xlabel(col, fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_ylabel('Count', fontsize=10)
        ax.tick_params(axis='y', labelsize=8)
        plt.show()

#Plot a Heatmap of Lisbon based on a variable
def plot_lisbon_heatmap(df, lat, long, variable):
    map_lisbon = folium.Map(location=[38.736946, -9.142685], zoom_start=12)

    # Add a heatmap layer to the map
    HeatMap(data= df[[lat, long, variable]], 
            radius=15, max_zoom=13, 
            gradient={0.2: 'blue', 0.4: 'purple', 0.6: 'pink', 0.8: 'orange', 1.0: 'red'}).add_to(map_lisbon)

    # Display the map
    return map_lisbon
    

# ------- EXPLORAÇÃO

#Find the most correlated based on a given threshold
def get_high_correlations(corr_matrix, threshold):
    corr_series = corr_matrix.stack()  # convert the correlation matrix into a series
    high_corr = corr_series[((corr_series > threshold) | (corr_series < -threshold)) & (corr_series < 1.0)]  # select pairs with correlation value higher than threshold
    high_corr = high_corr.reset_index()  # convert the series back to a dataframe
    high_corr.columns = ['Variable 1', 'Variable 2', 'Correlation']  # rename the columns
    high_corr = high_corr.sort_values('Correlation', ascending=False)  # sort by correlation value
    return high_corr

# -------- K-MEANS

#Find the optimum number of k based on inertia
def plot_inertia(data, k, times):
    fig, ax = plt.subplots()
    random_states = [np.random.randint(0, 1000) for i in range(times)]
    for i in random_states:
        inertia_kmeans = []
        for j in range(2, k):
            kmeans = KMeans(n_clusters=j, random_state=i).fit(data)
            inertia_kmeans.append(kmeans.inertia_)
        ax.plot(range(2, k), inertia_kmeans, 'x-', label=f'i={i}')
    ax.legend()
    plt.show()
