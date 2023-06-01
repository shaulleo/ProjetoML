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

# Umap library
import umap


#Association Rules
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


import warnings
import ast
warnings.filterwarnings("ignore")



colors_dict = {
    0: "#5D0000",
    1: "#39E34B",
    2: "#3981E3",
    3: "#FFC300",
    4: "#E339A8",
    5: "#5D00A7",
    6: "#88FFFB",
    7: "#FFC088",
    8: "#E1AFAF"}



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


def plot_histograms(df: pd.DataFrame, cols: list[str], hue_var = None) -> None:
    """
    Plots histograms using seaborn for specified columns of a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame with the data to plot.
    cols (list[str]): A list of strings representing the names of the columns to plot.
    hue_var (str): The column name representing the variable to color the plot.


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
        
        # #Plot the histogram for the column.
        fig, ax = plt.subplots(figsize=(12, 4))
        # sns.histplot(data= df, x=col, bins=bins, color='lightblue', ax=ax, hue=hue_var)

        # Determine the color for the histogram based on hue_var
        if hue_var is not None:
            hue_values = df[hue_var]
            colors = [colors_dict.get(value, 'grey') for value in hue_values]
            sns.histplot(data=df, x=col, bins=bins, ax=ax, hue=hue_var, palette=['blue', 'green', 'red'])
        else:
            sns.histplot(data=df, x=col, bins=bins, ax=ax, color='grey', hue=hue_var)
        

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

def pairplot(df: pd.DataFrame, cols: list[str], hue_var: str, sampling: Union[int, bool] = 5000) -> None:
    """
    Create a pairplot for the specified columns of a DataFrame, with optional sampling.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame with the data to plot.
    cols (list[str]): A list of strings representing the names of the columns to plot.
    hue_var (str): The column name representing the variable to color the plot.
    sampling (Union[int, bool], optional): The number of samples to include in the pairplot.
            If 0, no sampling is performed. Defaults to 5000.

    Returns:
        None
    """
    if sampling == 0:
        sns.pairplot(df[cols], hue = hue_var, kind = 'scatter', diag_kind = 'hist', corner = True, plot_kws = dict(alpha = 0.4), diag_kws=dict(fill=False), size = 3, palette = colors_dict)
        plt.show()
    else:
        sns.pairplot(df[cols].sample(sampling), hue = hue_var, kind = 'scatter', diag_kind = 'hist', corner = True, plot_kws = dict(alpha = 0.4), diag_kws=dict(fill=False), size = 3, palette = colors_dict)
        plt.show()


def boxplot_by(df: pd.DataFrame, cols: List[str], by_col: Union[str, None] = None) -> None:
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



def barplot_by(df: pd.DataFrame, cols: List[str], by_col) -> None:
    """
    Create barcharts of discrete variables grouped by another discrete variable.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame with the data to plot.
    cols (list[str]): A list of strings representing the names of the columns to plot.
    hue_var (str): The column name representing the variable to distinguish the barplots by.

    Returns:
        None
    """
    num_plots = len(cols)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10*num_rows))

    #Flatten the axs array for easy indexing
    axs = axs.flatten()

    for i, column in enumerate(cols):
        ax = axs[i]
        #df.groupby(column)[by_col].value_counts().unstack().plot(kind='bar', ax=ax, color=[colors_dict.get(x, 'gray') for x in df[cols].columns])
        plot_data = df.groupby(column)[by_col].value_counts().unstack()
        plot_data.plot(kind='bar', ax=ax, color=[colors_dict.get(x, 'gray') for x in plot_data.columns])
        ax.set_xlabel('')
        ax.set_title(column + ' by ' + by_col)

    for i in range(num_plots, len(axs)):
        axs[i].set_visible(False)

    #Adjust spacing between subplots
    fig.tight_layout()

    #Show the plot
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







#Association Rules -> só esta a funcionar para a o cluster_kmeansZ

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



