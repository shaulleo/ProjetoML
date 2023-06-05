#Basic Packages
import numpy as np
import pandas as pd
from datetime import date 
from typing import *

#Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

#To find addresses
from geopy.geocoders import Nominatim

#Visualization
import squarify
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Umap library
import umap


#Association Rules
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


import warnings
import ast
warnings.filterwarnings("ignore")



colors_gradient = {0 :'#dcf3dd',1 :'#c2f6c8',2 : '#8cf0ae',
                   3 : '#54edaa', 4 :'#0dd0c5',5: '#06829c',6 :'#035277',7: '#001c3e',8: '#00102b',
                   9: '#000c16', 10: '#00050a'}



colors_dict_g = {
    0: '#C9E6FF',
    1: '#B5CFE6',
    2: '#A1B8CC',
    3: '#8DA1B3',
    4: '#798A99',
    5: '#657380',
    6: '#515C66',
    7: '#3C454C',
    8: '#323A40',
    9: '#282E33',
    10: '#1E2326'
}

colors_dict = {0:  "#90a0de" , 1: "#ffb380", 2: "#e68aa5", 3: "#b07bdb" , 4: "#91e6c7",5: "#fce46d",6: "#b8b2b2"}


# ---------- DATA PRE-PROCESSING

def extract_education(row: str) -> int:
    """
    Extracts education level from a given customer name.

    Parameters:
    - row (str): A string representing the name of the customer.

    Returns:
    - education (int): An integer representing the education level of the customer:
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
    - row (str): A string representing the name of the customer.

    Returns:
    - name (str): A string representing the cleaned name of the customer.

    """
        #Split the names based on whitespace
    name_list = row.split(' ')
        #If the total number of names is superior to two, we assume there is an
            #education level associated
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
    - df (pd.DataFrame): A pandas DataFrame containing the birthdate column.
    - birthdate (str): A string representing the name of the birthdate column.

    Returns:
    - df (pd.DataFrame): A pandas DataFrame with the original data and three new 
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
    - row (pd.Series): A pandas series containing 'latitude' and 'longitude' 
                    columns representing the coordinates of the customer's address.

    Returns:
    - full address (str): A string representing the full address of the customer.

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
     - row (str): A string representing the full address.

     Returns:
     - str: A string representing the region of the address.

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
    - df (pd.DataFrame): A pandas DataFrame containing the data to be encoded.
    - lat (str): A string representing the name of the latitude column in the DataFrame.
    - long (str): A string representing the name of the longitude column in the DataFrame.
    - address (str): A string representing the name of the address column in the DataFrame.

    Returns:
    - df (pd.DataFrame): A pandas DataFrame with columns 'latitude_encoded' and 'longitude_encoded' 
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
    - df (pd.DataFrame): The pandas DataFrame to encode.
    - col (str): The name of the column to encode.
    - condition (bool): The condition used to encode the column.

    Returns:
    - df (pd.DataFrame): The pandas DataFrame with the specified column encoded.

    """
    #Replace the designated column by its encoding based on the provided condition.
    df[col] = np.where(condition, 0, 1)

    return df



def integer_convert(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Converts multiple columns of a pandas DataFrame to integer data type.

    Parameters:
    - df (pd.DataFrame): A pandas DataFrame.
    - cols (list[str]): A list of strings representing the names of the columns to convert.

    Returns:
    - df (pd.DataFrame): A pandas DataFrame with the columns converted to integer data type.

    """
    #Convert the columns into int64 data type.
    df[cols] = df[cols].astype('int64')

    return df



#------- VISUALIZATION


def plot_histograms(df: pd.DataFrame, cols: list[str], hue_var: str = None) -> None:
    """
    Plots histograms.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame with the data to plot.
    - cols (list[str]): A list of strings representing the names of the columns to plot.
    - hue_var (str): The column name representing the variable to color the plot. Default is None.


    Returns:
    - None

    """
    sns.set_style(style='white')

    #Define how the plots will be placed, based on the number of features being visualized.
    num_plots = len(cols)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 5*num_rows))

    #Flatten the axs array for easy indexing
    axs = axs.flatten()

    for i, col in enumerate(cols):
        #Compute the optimum number of bins using 
            # the Freedman–Diaconis rule.
        data__ = df[col]
        q75, q25 = np.percentile(data__, [75 ,25])
        iqr = q75 - q25
        bin_width = 2 * iqr * len(data__)**(-1/3)
        bins = int((data__.max() - data__.min()) / bin_width+0.1)
        ax = axs[i]
        #If there is a hue value:
        if hue_var is not None:
            sns.histplot(data=df, x=col, bins=bins, ax=ax, hue=hue_var, palette=colors_dict)
            ax.set_title(f'Histogram of {col} by {hue_var}', fontsize=14)
        else:
            sns.histplot(data=df, x=col, bins=bins, ax=ax, color='#5b9abd', linewidth=0.5, edgecolor=".2")
            ax.set_title(f'Histogram of {col}', fontsize=14)
        #Set title and labels.
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

    for i in range(num_plots, len(axs)):
        axs[i].set_visible(False)

    #Adjust spacing between subplots
    fig.tight_layout()

    plt.show()


def plot_bar_charts(df: pd.DataFrame, cols: List[str], by_col = None, invert_axis=True) -> None:
    """
    Plot bar charts, having the possibility of grouping them by another discrete variable.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame with the data to plot.
    - cols (list[str]): A list of strings representing the names of the columns to plot.
    - by_col (str): The name of the column representing the variable to group the bar charts by. Default is None.
    - invert_axis (bool): A boolean value indicating whether to invert the x-axis and y-axis. Default is True.
                True is applicable for the analysis by cluster.
    Returns:
    - None
    """

    #Define how the plots will be placed, based on the number of features being visualized.
    num_plots = len(cols)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 10*num_rows))

    #Flatten the axs array for easy indexing
    axs = axs.flatten()

    for i, column in enumerate(cols):
        ax = axs[i]
        #If grouping the data by a given column
        if by_col is not None:
            #If the analyzing by cluster (Want inverted axis)
            if invert_axis:
                #Define the data to be plotted
                
                plot_data = df.groupby(by_col)[column].value_counts().unstack()

                
                colors_gradient = {0 :'#f5fcf5',1 :'#c7fcce',2 : '#8cf0ae',
                   3 : '#54edaa', 4 :'#0dd0c5',5: '#06829c',6 :'#035277',7: '#001c3e',8: '#00102b',
                   9: '#000c16', 10: '#00050a', 11: '#00050a', 12: '#00050a', 13: '#00050a',14:'#00050a'}
                   
                #Plot the data with adjusted bar widths
                width = 0.8 / len(plot_data.columns)
                for j, col in enumerate(plot_data.columns):
                    counts = plot_data[col]
                    color = colors_gradient.get(col, '#6aa8cc')  # Get the corresponding color from the gradient
                    x = np.arange(len(counts))
                    ax.bar(x + j * width, counts.values, width=width, color=color, edgecolor='black')
                
                # Create a legend
                legend_handles = [mpatches.Patch(color=colors_gradient.get(col, '#6aa8cc'), label=col)
                                  for col in plot_data.columns]
                ax.legend(handles=legend_handles)

            else:
                #Plot the data
                plot_data = df.groupby(column)[by_col].value_counts().unstack()
                plot_data.plot(kind='bar', ax=ax, color=[colors_dict.get(x, '#6aa8cc') for x in plot_data.columns])

            ax.set_xlabel('')
            ax.set_title(column + ' by ' + by_col)
        #If there is no grouping
        else:
            #Plot a simple bar chart
            sns.countplot(x=column, data=df, ax=ax, color='#6aa8cc', linewidth=0.5, edgecolor=".2")
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8) 
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8) 

    for i in range(num_plots, len(axs)):
        axs[i].set_visible(False)

    #Adjust spacing between subplots
    fig.tight_layout()

    plt.show()


def stacked_bar_chart(df: pd.DataFrame, col: str, by_col: str):
    """
    Plot stacked bar charts of a discrete variabled grouped by another.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame with the data to plot.
    - col (str): The name of the column to plot..
    - by_col (str): The name of the column representing the variable to group the bar charts by.

    Returns:
    - None
    """
    #Group the data by the columns and calculate the count and the percentual count
    grouped_df = df.groupby([by_col, col]).size().unstack()
    percent_df = grouped_df.apply(lambda x: x / x.sum() * 100, axis=1)

    #Plot the stacked bar chart
    fig, ax = plt.subplots()
    percent_df.plot(kind='bar', stacked=True, ax=ax, color=[colors_dict.get(x, '#6aa8cc') for x in percent_df.columns])

    #Set the axis labels and title
    ax.set_xlabel(f'{by_col}')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Stacked Bar Chart of {col} by {by_col}')

    #Rotate the x-axis tick labels to be vertical
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    #Show the legend
    ax.legend()

    #Display the chart
    plt.show()


def plot_violinplot(df: pd.DataFrame, cols: list[str], by_col: str) -> None:
    """
    Plots a set of violin plots of given list of variables against one specific variable.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame with the data to plot.
    - cols (list[str]): A list of strings representing the names of the columns to plot.
    - by_col (str): The column name representing the variable to distinguish the Violin plots by.

    Returns:
    - None
    """
    sns.set_style(style='white')
    # Define how the plots will be placed, based on the number of features being visualized.
    num_plots = len(cols)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 5*num_rows))

    # Flatten the axs array for easy indexing
    axs = axs.flatten()

    for i, col in enumerate(cols):
        ax = axs[i]
        sns.violinplot(x=by_col, y=col, data=df, ax=ax, color='lightblue')
        ax.set_title(col.capitalize())

    plt.tight_layout()
    plt.show()


def plot_lisbon_heatmap(df: pd.DataFrame, lat: str, long: str, col: str) -> folium.folium.Map:
    """
    Plot a heatmap of the data, based on a specific variable, centered in Lisbon.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the latitude, longitude, and variable data.
    - lat (str): A string representing the name of the latitude column in the DataFrame.
    - long (str): A string representing the name of the longitude column in the DataFrame.
    - col (str): The name of the column in the dataframe containing the  data to be represented.

    Returns:
    - map_lisbon (folium.folium.Map): A folium Map object displaying the heatmap.
    """
        #Create a folium map centered in Lisbon.
    map_lisbon = folium.Map(location=[38.736946, -9.142685], zoom_start=12)

        #Create a Heatmap layer for the specified column with a gradient.
    HeatMap(data=df[[lat, long, col]], 
            radius=15, max_zoom=13, 
            gradient={0.2: 'blue', 0.4: 'purple', 0.6: 'pink', 0.8: 'orange', 1.0: 'red'}).add_to(map_lisbon)

    return map_lisbon



def regional_treemap(df: pd.DataFrame) -> None:
    """
    Plots a Treemap visualization of the number of observations by region.

    Parameters:
    - df (pd.DataFrame): A pandas DataFrame containing the data 
        grouped by region, including the counts and the region itself.
    Returns:
    - None
    """
    sns.set_style(style="whitegrid") 
    sizes = df["count"].values 
    labels = [f"{r}\n({c})" for r, c in zip(df["region"], df["count"])]

    #Sort sizes and labels in descending order
    sorted_indices = sorted(range(len(sizes)), key=lambda k: sizes[k], reverse=True)
    sorted_sizes = [sizes[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    #Create a dark-to-light color palette based on sorted sizes
    palette = sns.color_palette("Blues", n_colors=len(labels))
    palette = list(reversed(palette))

    plt.figure(figsize=(18, 10))

    #Plot the tree map using squarify
    squarify.plot(sizes=sorted_sizes, label=sorted_labels, alpha=0.8, color=palette, text_kwargs={'fontsize': 10})
    plt.title('Observations by Region', fontsize=16)
    plt.axis('off')
    plt.show()



def pairplot(df: pd.DataFrame, cols: list[str], hue_var: str = None, sampling: int = 5000, data_type: str = 'continuous', transparency: float =0.4) -> None:
    """
    Create a pairplot for the specified columns of a DataFrame, with optional sampling.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame with the data to plot.
    - cols (list[str]): A list of strings representing the names of the columns to plot.
    - hue_var (str): The column name representing the variable to color the plot.
    - sampling (int): The number of samples to use when creating the pairplot.
            If 0, all data is used. Defaults to 5000.
    - data_type (str): The type of data being plotted, either 'continuous' or 'discrete', such that
            the output is a scatterplot or a kdeplot, respectively. Defaults to 'continuous'.
    - transparency (float): The transparency level of the plotted points. Defaults to 0.4.

    Returns:
    - None
    """

    sns.set_style(style='white')

    #If the features being used are continuos, plot the scatterplot and histograms on the pairplot.
    if data_type == 'continuous':
        #If using sampling or not
        if sampling == 0:
            sns.pairplot(df[cols], hue = hue_var, kind = 'scatter', diag_kind = 'hist', corner = True, plot_kws = dict(alpha = transparency), diag_kws=dict(fill=False), size = 3, palette = colors_dict)
            plt.show()
        else:
            sns.pairplot(df[cols].sample(sampling), hue = hue_var, kind = 'scatter', diag_kind = 'hist', corner = True, plot_kws = dict(alpha = transparency), diag_kws=dict(fill=False), size = 3, palette = colors_dict)
            plt.show()
    #If the features being used are discrete, plot the KDEplots on the pairplot.
    elif data_type == 'discrete':
        if sampling == 0:
            sns.pairplot(df[cols], hue = hue_var, kind = 'kde', diag_kind = 'kde', corner = True, plot_kws = dict(alpha = transparency), diag_kws=dict(fill=False), size = 3, palette = colors_dict)
            plt.show()
        else:
            sns.pairplot(df[cols].sample(sampling), hue = hue_var, kind = 'kde', diag_kind = 'kde', corner = True, plot_kws = dict(alpha = transparency), diag_kws=dict(fill=False), size = 3, palette = colors_dict)
            plt.show()



def boxplot_by(df: pd.DataFrame, cols: list[str], by_col: Union[str, None] = None) -> None:
    """
    Create boxplots of continuous variables grouped by a discrete variable.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame with the data to plot.
    cols (list[str]): A list of strings representing the names of the columns to plot.
    hue_var (str): The column name representing the variable to distinguish the boxplots by.

    Returns:
        None
    """
    num_plots = len(cols)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 10*num_rows))

    #Flatten the axs array for easy indexing
    axs = axs.flatten()

    for i, column in enumerate(cols):
        ax = axs[i]
        df.boxplot(column=column, by=by_col, ax=ax)
        ax.set_xlabel('')
        ax.set_title(column + ' by ' + by_col )
        
    for i in range(num_plots, len(axs)):
        axs[i].set_visible(False)

    #Adjust spacing between subplots
    fig.tight_layout()

    plt.show()



def scatterplot(df: pd.DataFrame, cols:list[str], by_col: str) -> None:
    """
    Plots a set of scatterplots of given list of variables against one specific variable.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame with the data to plot.
    - cols (list[str]): A list of strings representing the names of the columns to plot.
    - by_col (str): The column name that represents the y axis.


    Returns:
    - None
    """
    sns.set_style(style='white')
    #Define how the plots will be placed, based on the number of features being visualized.
    num_plots = len(cols)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 5*num_rows))

    #Flatten the axs array for easy indexing
    axs = axs.flatten()

    for i, col in enumerate(cols):
        ax = axs[i]
        sns.scatterplot(x=col, y= by_col, data= df, ax=ax, alpha = 0.2)
        ax.set_title(col.capitalize())
    
    plt.tight_layout()
    
    plt.show()



# ------- EXPLORATION

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
    df (pd.DataFrame): A DataFrame containing the data with the 
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



def silhoette_method(df: pd.DataFrame, cluster_col: str) -> None:
    """
    Compute and visualize the Silhouette method for evaluating the quality of a clustering solution.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the data with the 
    designated clusters.
    cluster_col (str): The name of the column containing the cluster information.

    Returns:
        None: Displays the Silhouette plot and prints the Silhouette score.
    """
    #Access the cluster labels 
    cluster_labels = df[cluster_col]

    #Specify the number of clusters
    n_clusters = len(cluster_labels.unique())
    #n_clusters = 6

    #Calculate silhouette score for the clustering solution
    silhouette_avg = silhouette_score(df, cluster_labels)

    #Compute silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df, cluster_labels)

    #Plot silhouette visualization
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 4)

    y_lower = 10
    for i in range(n_clusters):
        #Aggregate silhouette scores for samples in the current cluster
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        #Sort the silhouette scores in descending order
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        #Color the clusters
        #color = plt.cm.get_cmap("Spectral")(float(i) / n_clusters)
        color = colors_dict[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        #Label each cluster silhouette plot with the cluster number
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        #Compute the new y_lower for the next plot
        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    #The vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])
    ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_title("Silhouette plot for {} clusters".format(n_clusters))

    plt.show()

    #Print the silhouette score
    print("Silhouette score for {} clusters: {:.4f}".format(n_clusters, silhouette_avg))



def umap_plot(df: pd.DataFrame, cluster_col: str) -> None:
    """
    Plot the UMAP embedding of a DataFrame, colouring by cluster.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the data with the 
    designated clusters.
    cluster_col (str): The name of the column containing the cluster information.

    Returns:
        None
    """
    #Fit the umap
    reducer = umap.UMAP(random_state=42)

    #Extract cluster labels
    labels = df[cluster_col].values
    n_clusters = df[cluster_col].nunique()
    #Compute the UMAP embedding
    embedding = reducer.fit_transform(df)
    
    #Create a colormap with a varying number of colors based on the number of clusters
    #cmap = plt.cm.get_cmap('viridis', n_clusters)
    
    #plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[colors_dict[label] for label in labels])
    plt.gca().set_aspect('equal', 'datalim')
    
    # Create a colormap using the colors from the dictionary
    cmap = mcolors.ListedColormap([colors_dict[label] for label in range(n_clusters)])
    norm = mcolors.BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)

    # Create a colorbar with the colors and labels from the dictionary
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm))
    cbar.set_ticks(np.arange(n_clusters))
    cbar.set_ticklabels([f"{label}" for label in range(n_clusters)])

    plt.show()


# ------- ASSOCIATION RULES


def preprocess_basket(df: pd.DataFrame, cluster: int) -> pd.DataFrame:
    """
    Preprocess the basket data for a specific cluster.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the transactional data with the 
    respetive customer's clusters.
    - cluster (int): The cluster number.

    Returns:
    - transaction_items (pd.DataFrame): A dataframe with the items for each transaction
    of the given cluster.
    """

    #Filter basket data for the specified cluster
    filtered_basket = df[df['segment'] == cluster]

    #Drop unnecessary columns
    filtered_basket.drop(['customer_id','segment'], inplace=True, axis=1)

    #Convert the 'list_of_goods' column of the basket values to lists
    filtered_basket = [ast.literal_eval(element) for element in list(filtered_basket['list_of_goods'])]
    te = TransactionEncoder()
    te_fit = te.fit(filtered_basket).transform(filtered_basket)

    #Perform one-hot encoding
    transaction_items = pd.DataFrame(te_fit, columns= te.columns_)
    
    return transaction_items



def build_rules(df: pd.DataFrame, min_support: float, metric: str, min_threshold: float) -> pd.DataFrame:
    """
    Build association rules from frequent itemsets.

    Parameters:
    - df (pd.DataFrame): A one-hot encoded dataframe with the items for each transaction.
    - min_support (float): The minimum support threshold for finding frequent item sets.
    - metric (str): The metric used to evaluate the association rules.
    - min_threshold (float): The minimum threshold of the metric that generates the 
    association rules.

    Returns:
    - rules (pd.DataFrame): DataFrame containing the generated association rules.
    """
    #Find frequent item groups
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    #Generate association rules based on a given metric and threshold
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    
    return rules

