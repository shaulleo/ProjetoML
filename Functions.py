#Functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Para encontrar a morada de cada
import reverse_geocoder as rg
from geopy.geocoders import Nominatim

from datetime import date 


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


#Retirar local de morada
def get_address_1(row):
    geolocator = Nominatim(user_agent='my_app')
    full_address = geolocator.reverse(f"{row['latitude']}, {row['longitude']}").address
    full_address = full_address.split(',')
    return full_address[-4]

def get_address_2(row): 
    geolocator = Nominatim(user_agent='my_app') 
    full_address = geolocator.reverse(f"{row['latitude']}, {row['longitude']}").address 
    #full_address = full_address.split(',') 
    return full_address

def get_address(row): 
    geolocator = Nominatim(user_agent='my_app') 
    full_address = geolocator.reverse(f"{row['latitude']}, {row['longitude']}").address 
    return full_address


#Limpar o endereço num só 
#Função Shaul
def clean_address(row):
    full_address = row.split(',')
    if len(full_address) > 4:
        address = full_address[-4]
    else:
        address = full_address[-3]
    if address[0] == " ":
        address = address[1:]
    return address

#Função Bruno
def clean_address_bruno(row):
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


#A partir da freguesia extrai no cluster o valor medio de lat/long e faz o encoding respectivamente
def encode_address(dataframe, latitude, longitude, address):
    lat_map = dataframe.groupby(address)[latitude].mean().to_dict()
    long_map = dataframe.groupby(address)[longitude].mean().to_dict()
    dataframe['latitude_encoded'] = dataframe[address].map(lat_map)
    dataframe['longitude_encoded'] = dataframe[address].map(long_map)
    return dataframe



# Função que faz o encoding não binário (Madalena)
def categorical_encoding(df, col_name, replace_dict):
    new_col_name = col_name + '_encoded'
    df[new_col_name] = df[col_name].map(replace_dict)
    # df = df.drop(col_name, axis=1) # faz o drop da coluna inicial
    return df

# Função que faz o encoding binário
#Se a condição for true, a col_name tem valor 0, caso contrário tem 1
def binary_encoding(df, col_name, condition):
    #new_col_name = col_name + '_encoded'
    df[col_name] = np.where(condition, 0, 1)
    return df

#Fazer função para convert varias cols de uma dataframe em int
def integer_convert(df, cols):
    # Converte as colunas especificadas em cols de float para int
    df[cols] = df[cols].astype('int64')
    return df



#------- VISUALIZAÇÃO

#Plot histograms based on optimum number of bins
def plot_histogram(df, column_name):
    # extract the data from the specified column
    data = df[column_name]
    
    # calculate the optimal number of bins using the Freedman-Diaconis rule
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data)**(-1/3)
    num_bins = int((data.max() - data.min()) / bin_width)
    
    # plot the histogram with the optimal number of bins
    plt.hist(data, bins=num_bins)
    
    # set the plot title and axis labels
    plt.title('Histogram of ' + column_name)
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    
    # display the plot
    plt.show()


#Plot histograms (com seaborn - inspirado nos gráficos do Bruno)
def seaborn_histograms(df, column_name):
    sns.histplot(df[column_name], stat = 'count', bins= 'auto').set(title=column_name)


#Função para os countplots
def bar_charts(df, var):
    sns.countplot(data=df, x=var)
    plt.show()