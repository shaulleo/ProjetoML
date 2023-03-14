#Functions

#Extract education level from customer name
def extract_education(observation):
    name_list = observation.split(' ')
    if len(name_list) > 2:
        if name_list[0] == 'Msc.':
            education = 'Masters Degree'
        elif name_list[0] == 'Bsc.':
            education = 'Bachelor Degree'
        else:
            education = 'Doctoral Degree'
    else:
        education = 'Basic'
    return education


#Clean the customers name
def clean_names(observation):
    name_list = observation.split(' ')
    if len(name_list) > 2:
        name = str(name_list[1] +' '+ name_list[2])
    else:
        name = str(name_list[0] + ' '+ name_list[1])
    return name


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


#Separate birthday date into three different columns
#-- observation vem em formato TimeStamp
def birthday(observation):
    date = str(observation).split(' ')[0]
    date = date.split('-')
    day = int(date[0])
    month = int(date[1])
    year = int(date[2])
    return day, month, year


#Descobrir signo zodíaco :P
