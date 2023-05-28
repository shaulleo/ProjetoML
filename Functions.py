#Basic Packages
import numpy as np
import pandas as pd
from datetime import date 

#Clustering
from sklearn.cluster import KMeans

#To find addresses
from geopy.geocoders import Nominatim

#Visualization
import squarify
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

#Association Rules
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


import warnings
import ast
warnings.filterwarnings("ignore")

#Functions

# ---------- DATA PRE-PROCESSING

def extract_education(row: str) -> int:
    """
    Extracts education level from a given customer name.

    Parameters:
    row (str): A string representing the name of the customer.

    Returns:
    int: An integer representing the education level of the customer:
         1: If the customer has a BSc. degree
         2: If the customer has a MSc. degree
         3: If the customer does not have a degree
         0: If the customer's name is not sufficient to 
         determine their education level

    """
        #Split the name into a list based on whitespace.
    name_list = row.split(' ')
        #If the length of the list is higher than two, 
        # determine the education level. 
    if len(name_list) > 2:
        if name_list[0] == 'Msc.':
            education = 2
        elif name_list[0] == 'Bsc.':
            education = 1
        else:
            education = 3
        #Otherwise, assume education level as 0.
    else:
        education = 0
    
    return education



def clean_names(row: str) -> str:
    """
    Cleans a customer name by removing education level titles and 
    returning only their first and last name.

    Parameters:
    row (str): A string representing the name of the customer.

    Returns:
    str: A string representing the cleaned name of the customer.

    """
    name_list = row.split(' ')
    if len(name_list) > 2:
        name = str(name_list[1] + ' ' + name_list[2])
    else:
        name = str(name_list[0] + ' ' + name_list[1])
    
    return name



def process_birthdate(df: pd.DataFrame, birthdate: str) -> pd.DataFrame:
    """
    Separates the customer's birthdate column into birthday and birthmonth 
    columns, and calculate their age.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the birthdate column.
    birthdate (str): A string representing the name of the birthdate column.

    Returns:
    pd.DataFrame: A pandas DataFrame with the original data and three new 
    columns for birthday and birthmonth as well as a column for age.

    """
        #Convert the birthdate column into a pd.Datetime object.
    df[birthdate] = pd.to_datetime(df[birthdate])
        #Extract the day of the birthdate column.
    df['birthday'] = df[birthdate].dt.day
        #Extract the month of the birthdate column.
    df['birthmonth'] = df[birthdate].dt.month
        #Extract the present year.
    year_ = date.today().year 
        #Compute the age based on the present year and the birthdate column.
    df['age'] = int(year_) - df[birthdate].dt.year

    return df



def get_address(row: pd.Series) -> str:
    """
    Retrieves the customer's full address based on their given latitude and longitude values.

    Parameters:
    row (pd.Series): A pandas series containing 'latitude' and 'longitude' 
                    columns representing the coordinates of the customer's address.

    Returns:
    str: A string representing the full address of the customer.

    """
        #Create a Nominatim geocoding instance.
    geolocator = Nominatim(user_agent='my_app') 
        #Find the full address based on the address cordinates.
    full_address = geolocator.reverse(f"{row['latitude']}, {row['longitude']}").address 
    
    return full_address



def clean_address(row: str) -> str:
    """
     Extract the region information from the full address.

     Parameters:
     row (str): A string representing the full address.

     Returns:
     str: A string representing the region of the address.

    """
    #Split the full address into a list based on comma separators.
    full_address = row.split(',')
    #Verify if the list length is lower than 4. If so, assign the address as the 
    #third last value in the list, otherwise return the fourth last value in the list,
    #if Lisbon is contained in the region name.
    address = full_address[-3] if len(full_address) < 4 else full_address[-4] if 'Lisboa' in full_address[-3] else full_address[-3]

    #Cleanse the region name, in the case of it beginning with a whitespace.
    return address.strip()



def encode_address(df: pd.DataFrame, lat: str, long: str, address: str) -> pd.DataFrame:
    """
    Encodes the address by extracting the mean value of latitude and longitude 
    of the data for every region.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the data to be encoded.
    lat (str): A string representing the name of the latitude column in the DataFrame.
    long (str): A string representing the name of the longitude column in the DataFrame.
    address (str): A string representing the name of the address column in the DataFrame.

    Returns:
    pd.DataFrame: A pandas DataFrame with columns 'latitude_encoded' and 'longitude_encoded' 
    added to it, representing the encoded coordinates of the observation.

    """
        #Create a dictionary which represents the mean values of latitude 
        # for every region.
    lat_map = df.groupby(address)[lat].mean().to_dict()
        #Create a dictionary which represents the mean values of longitude 
        # for every region.
    long_map = df.groupby(address)[long].mean().to_dict()
        #Map the values of the lat_map dictionary to the dataframe based 
        # on the address.
    df['latitude_encoded'] = df[address].map(lat_map)
        #Map the values of the lat_map dictionary to the dataframe based 
        # on the address.
    df['longitude_encoded'] = df[address].map(long_map)

    return df



def binary_encoding(df: pd.DataFrame, col: str, condition: bool) -> pd.DataFrame:
    """
    Encodes binary values in a pandas DataFrame based on a condition.

    If the condition is True, the specified column in the DataFrame is set to 0,
    otherwise it is set to 1.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame to encode.
    col (str): The name of the column to encode.
    condition (bool): The condition used to encode the column.

    Returns:
    pd.DataFrame: The pandas DataFrame with the specified column encoded.

    """
    #Replace the designated column by its encoding based on the provided condition.
    df[col] = np.where(condition, 0, 1)

    return df



def integer_convert(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Converts multiple columns of a pandas DataFrame to integer data type.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame.
    cols (list[str]): A list of strings representing the names of the columns to convert.

    Returns:
    pd.DataFrame: A pandas DataFrame with the columns converted to integer data type.

    """
    #Convert the columns into int64 data type.
    df[cols] = df[cols].astype('int64')

    return df



#------- VISUALIZATION



#AJUSTAR DOCSTRINGS C BASE NA HUE
def plot_histograms(df: pd.DataFrame, cols: list[str], hue_var = None) -> None:
    """
    Plots histograms using seaborn for specified columns of a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame with the data to plot.
    cols (list[str]): A list of strings representing the names of the columns to plot.


    Returns:
    None.

    """
    #Loop over each column.
    for col in cols:
        data__ = df[col]
            #Determine the optimal number of bins based on the data type of the column.
            #If the data type is an integer, assign the number of bins as the number of 
            # unique values within the data.
        if data__.dtype == 'int64':
            bins = data__.nunique()
            #If the data type is float, compute the optimum number of bins using 
            # the Freedman–Diaconis rule.
        elif data__.dtype == 'float64':
            q75, q25 = np.percentile(data__, [75 ,25])
            iqr = q75 - q25
            bin_width = 2 * iqr * len(data__)**(-1/3)
            bins = int((data__.max() - data__.min()) / bin_width)
            #Otherwise, do not calculate the number of bins.
        else:
            print(f"Skipping column '{col}' - not a numeric datatype")
            continue
        
        #Plot the histogram for the column.
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.histplot(data= df, x=col, bins=bins, color='lightblue', ax=ax, hue=hue_var)
        #Set title and labels.
        ax.set_title(f'{col} histogram', fontsize=14)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        plt.show()



def plot_bar_charts(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Plots bar charts using seaborn for specified columns of a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame with the data to plot.
    cols (list[str]): A list of strings representing the names of the columns to plot.

    Returns:
    None
    """
    #Loop over each column:
    for col in cols:
        #Plot the bar chart for the column.
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.countplot(x=col, data=df, ax=ax, color='lightblue', linewidth=1, edgecolor=".2")

        #Set title and labels.
        ax.set_title(f'{col}', fontsize=12)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)

        #Set font sizes for the axis.
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        plt.show()


#FUNÇÃO MADALENA SPLIT HISTOGRAMA
def plot_split_histogram(data, split_value):
    split_data1 = data[data <= split_value]
    split_data2 = data[data > split_value]

    q75_1, q25_1 = np.percentile(split_data1, [75 ,25])
    iqr_1 = q75_1 - q25_1
    bin_width_1 = 2 * iqr_1 * len(split_data1)**(-1/3)
    bins_1 = int((split_data1.max() - split_data1.min()) / bin_width_1)

    q75_2, q25_2 = np.percentile(split_data2, [75 ,25])
    iqr_2 = q75_2 - q25_2
    bin_width_2 = 2 * iqr_2 * len(split_data2)**(-1/3)
    bins_2 = int((split_data2.max() - split_data2.min()) / bin_width_2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    sns.histplot(x=split_data1, bins=bins_1, color='lightblue', ax=ax1)
    ax1.set_title(f'{data.name} histogram (0-{split_value})', fontsize=14)
    ax1.set_xlabel(data.name, fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_xlim([0, split_value])

    sns.histplot(x=split_data2, bins=bins_2, color='lightblue', ax=ax2)
    ax2.set_title(f'{data.name} histogram ({split_value}-{data.max()})', fontsize=14)
    ax2.set_xlabel(data.name, fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_xlim([split_value, data.max()])

    plt.subplots_adjust(hspace=0.5)
    plt.show()

# o unico problema aqui é o facto dos eixos do y não estarem iguais para os dois gráficos, mas a função esta feita para que seja possivel 
# escolhr onde queremos fazer o split


def plot_lisbon_heatmap(df: pd.DataFrame, lat: str, long: str, col: str) -> folium.folium.Map:
    """
    Plot a heatmap of the data, based on a specific variable, centered in Lisbon.

    Parameters:
    df (pd.DataFrame): The dataframe containing the latitude, longitude, and variable data.
    lat (str): A string representing the name of the latitude column in the DataFrame.
    long (str): A string representing the name of the longitude column in the DataFrame.
    col (str): The name of the column in the dataframe containing the  data to be represented.

    Returns:
    folium.folium.Map: A folium Map object displaying the heatmap.
    """
        #Create a folium map centered in Lisbon.
    map_lisbon = folium.Map(location=[38.736946, -9.142685], zoom_start=12)

        #Create a Heatmap layer for the specified column with a gradient.
    HeatMap(data=df[[lat, long, col]], 
            radius=15, max_zoom=13, 
            gradient={0.2: 'blue', 0.4: 'purple', 0.6: 'pink', 0.8: 'orange', 1.0: 'red'}).add_to(map_lisbon)

    return map_lisbon


def regional_treemap(df):
    sns.set_style(style="whitegrid") 
    sizes= df["count"].values 
    label=df["region"]
    squarify.plot(sizes=sizes, label=label, alpha=0.6).set(title='Observations by Region')
    plt.axis('off')
    sns.set(rc={'figure.figsize':(17,17)})
    plt.show()


# ------- EXPLORAÇÃO

def get_high_correlations(corr_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Find the pairs of variables that have a correlation value higher than a given threshold.

    Parameters:
    corr_matrix (pd.DataFrame): The correlation matrix to be analyzed.
    threshold (float): The correlation threshold value.

    Returns:
    pd.DataFrame: A dataframe containing the pairs of variables whose correlation is above 
    the given threshold and the respective correlation value.
    """
        #Create a mask to only show the lower triangle of the correlation matrix.
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        #Apply the mask to the correlation matrix and store it.
    lower_triangle_corr = corr_matrix.where(mask)
        #Convert the lower triangle of the correlation matrix into a series.
    corr_series = lower_triangle_corr.stack()
        #Select the pairs whose correlation value is higher than the given threshold.
    high_corr = corr_series[((corr_series > threshold) | (corr_series < -threshold)) & (corr_series < 1.0)] 
        #Convert the series back to a dataframe.
    high_corr = high_corr.reset_index()  
        #Rename the columns.
    high_corr.columns = ['Variable 1', 'Variable 2', 'Correlation']
        #Sort the dataframe by the correlation values, in descending order.
    high_corr = high_corr.sort_values('Correlation', ascending=False)
    
    return high_corr


def group_by_region(df: pd.DataFrame, region_col: str, cols: list[str]):
    """
    Find the mean values of given set of columns and the count of observations by region.

    Parameters:
    df (pd.DataFrame): Input DataFrame to group by region.
    region_col (str): A string representing the name of the column to group by (the region column).
    cols (list[str]): List of column names to calculate the mean values for.

    Returns:
    pd.DataFrame: DataFrame containing the mean value of the specified columns
    per region and the count of observations per region.
    """
    nr_region = df.groupby(region_col)[cols[0]].count().reset_index()
    nr_region.rename(columns={cols[0]: 'count'}, inplace=True)

    # Group the data by region and calculate the average total lifetime and others spend per region
    region_data = df.groupby(region_col)[cols].mean()

    region_data = region_data.merge(nr_region, on=region_col)
    
    return region_data



# -------- K-MEANS

def plot_inertia(df: pd.DataFrame, k: int, times: int) -> None:
    """
    Plot the inertia of the K-Means algorithm for different values of k a given 
    number of times, to find the optimum number of k.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data aimed to be clustered.
    k (int): The maximum number of clusters to try.
    times (int): The number of times to run K-Means for each value of k.

    Returns:
    None
    """

    fig, ax = plt.subplots()
        #Generate a list of random states for the K-Means algorithm.
    random_states = [np.random.randint(0, 1000) for i in range(times)]
        #Iterate over the random states.
    for i in random_states:
        #Create an empty list to store the inertia values.
        inertia_kmeans = []
        #Iterate over the number of clusters to try.
        for j in range(2, k):
            #Run the K-Means algorithm and store the inertia value in 
            # the designated list.
            kmeans = KMeans(n_clusters=j, random_state=i).fit(df)
            inertia_kmeans.append(kmeans.inertia_)
        #Plot the inertia values for every random state.
        ax.plot(range(2, k), inertia_kmeans, 'x-', label=f'i={i}')
    
    ax.legend()
    plt.show()


def compare_clusters(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """
    Creates a DataFrame with the mean value of each column for different 
    clusters and for all observations.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame containing the data with the 
    designated clusters.
    cluster_col (str): The name of the column containing the cluster information.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the mean values of each column, 
    grouped by the cluster column and the general mean for each variable.

    """
    #Compute the mean of the variable without segmentation
    general_mean = pd.DataFrame(df.mean().T).rename(columns={0:'general_mean'})
    #Find mean values of the variable per cluster
    clusters_mean = pd.DataFrame(df.groupby(cluster_col).mean().T)
    
    return clusters_mean.join(general_mean)


#Association Rules

def preprocess_basket(df, cluster):
    filtered_basket = df[df['cluster_kmeansZ'] == cluster]
    filtered_basket.drop(['customer_id','cluster_kmeansZ'], inplace=True, axis=1)
    filtered_basket = [ast.literal_eval(element) for element in list(filtered_basket['list_of_goods'])]
    te = TransactionEncoder()
    te_fit = te.fit(filtered_basket).transform(filtered_basket)
    transaction_items = pd.DataFrame(te_fit, columns= te.columns_)
    return transaction_items

def build_rules(df, min_support, metric, min_threshold):
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return rules



