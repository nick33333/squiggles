import os
# Data thingy libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram

from sklearn.metrics import pairwise_distances
# Algorithms
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import FeatureAgglomeration
from scipy import interpolate
# from tslearn
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import dtw
from tslearn.metrics import soft_dtw
from tslearn.clustering import KShape
from tslearn.clustering import TimeSeriesKMeans


def log10_with_zero(x):
    '''
    np.log10(0) kept returning -inf (probably for good reason), but it kept
    being annoying so I asked chatGPT to make a log10 func to return 0 where
    np.log10 returns -inf because one time point altered with this shouldn't
    affect the clustering too bad I hope.
    '''
    return np.where(x == 0, 0, np.log10(x))


def hierarchical_clustering(distance_matrix, method='complete'):
    '''
    Not really used or touched on
    Function from https://towardsdatascience.com/how-to-apply-hierarchical-clustering-to-time-series-a5fe2a7d8447
    '''
    if method == 'complete':
        Z = complete(distance_matrix)
    if method == 'single':
        Z = single(distance_matrix)
    if method == 'average':
        Z = average(distance_matrix)
    if method == 'ward':
        Z = ward(distance_matrix)
    fig = plt.figure(figsize=(16, 8))
    dn = dendrogram(Z)
    plt.title(f"Dendrogram for {method}-linkage with correlation distance")
    plt.show()
    return Z


def select_table_subset_by_label(MC, label):
    '''
    Given a clustering table cols(Latin name, Sample, Labels) and a desired
    label, return a subset of the clustering table with the desired label.
    '''
    return MC.clusterTable.loc[MC.clusterTable["Labels"] == label]


def label2seriesMatrix(MC, label):
    '''
    Given a clustering table cols(Latin name, Sample, Labels) and a desired
    label, return a distance matrix of the samples marked with desired label.
    '''
    series_names = select_table_subset_by_label(MC, label).Sample.tolist()
    series_list = [MC.name2series[series_name] for series_name in series_names]
    distance_matrix = np.zeros((len(series_list), len(series_list)))
    for idx, i in enumerate(series_list):
        for jdx, j in enumerate(series_list):
            # Calculate pairwise distances using dynamic time warping
            distance = dtw(i.to_numpy(),
                           j.to_numpy())
            distance_matrix[idx][jdx] = distance
            distance_matrix[jdx][idx] = distance
    return distance_matrix, series_names


def max_pairwise_dists_of_cluster(MC):
    '''max_pairwise_dists_of_cluster
    Finds the max pairwise distances observed in each cluster
    '''
    label_maxdists_lite = dict()
    label_maxdists = dict()
    for label in sorted(list(set(MC.dtw_labels))):
        # Create cluster dist matrix
        dm_label, names_label = label2seriesMatrix(MC, label)
        # Find Max pairwise distance within dist matrix of cluster
        idx = np.unravel_index(np.argmax(dm_label), dm_label.shape)  # finds index of max val in matrix
        label_maxdists[label] = (dm_label[idx], dm_label, names_label)
        label_maxdists_lite[label] = dm_label[idx]
    return label_maxdists_lite, label_maxdists


# Define the distance metric to be DTW distance
def dtw_distance(X, Y=None, **kwargs):
    if Y is None:
        return 0.0
    else:
        return dtw(X, Y)


def softdtw_distance(X, Y=None, gamma=1.0, **kwargs):
    if Y is None:
        return 0.0
    else:
        return soft_dtw(X, Y)


def seriesWindow(df, by, upperbound, lowerbound):
    '''
    Truncates pandas df to desired start and stop bounds by a designated
    field
    '''
    index = (df[by] < upperbound) & (df[by] > lowerbound)
    return df.loc[index]


def windowMySeries(mySeries, namesofMySeries, **kwargs):
    '''
    seriesWindow wrapper function.
    Takes in a list of time series dataframes (and names list) and returns a
    truncated version of that list of time series dataframes. Truncation is
    performed with seriesWindow function which takes in dataframes, a field to
    truncate by, an upperbound on the field, and a lowerbound on the field.
    '''
    windowedSeries = []
    windowedNamesofMySeries = [] # We will want to select only the names of samples which aren't omitted by windowing.
    for idx, df in enumerate(mySeries):
        windowed_df = seriesWindow(df, **kwargs)
        if len(windowed_df) > 0:  # If datapoints remain in df after windowing it by real time
            windowedSeries.append(windowed_df) # Append windowed time series df
            windowedNamesofMySeries.append(namesofMySeries[idx]) # Append name of windowed time series df
    return windowedSeries, windowedNamesofMySeries


def normalize_series_column(series_column: "pd.series") -> "pd.series":
    '''
    Normalize a pandas series_column (df or col) to [0, 1].
    Handle case where all data may be 0
    Kinda lame, but self.flat_curves this is the only reason
    log10_scale_time_and_normalize_values and normalize_series_column are methods
    '''
    if len(series_column.unique()) == 1:
        flat_curve = 1
        return np.zeros(len(series_column)), flat_curve
    else:
        flat_curve = 0
        return (series_column - series_column.min()) / (series_column.max() - series_column.min()), flat_curve


def log10_scale_time_and_normalize_values(mySeries,
                                          namesofMySeries,
                                          use_time_log10_scaling,
                                          use_value_normalization,
                                          time_field,
                                          value_field,
                                          ):
    '''
    Transform time column into log10 scale [BAD]
        - Due to use of DTW, using a log10 transform exponentially decreases
          the difference (time) between data points as ts curve will be in terms
          of magnitude rather than time
        - Instead, maybe forget about log10 transform except when it comes to plotting
    Don't 
    and either
    (1) Normalize lambda column to [0, 1]
    (2) Divide lambda column by 1e4
    '''
    newMySeries = []
    newNamesOfMySeries = []
    flat_curves = 0
    for idx, series in enumerate(mySeries):
        if len(series[value_field]) == 1: # This is only kind of redundant so I'll keep it lol
            flat_curve = 1
        else:
            flat_curve = 0
        if use_time_log10_scaling:
            series[time_field] = log10_with_zero(series[time_field])
        if use_value_normalization:
            series[value_field], flat_curve = normalize_series_column(series[value_field]) # Where flat_curve's redudancy shows
        if flat_curve == 0:
            newMySeries.append(series)
            newNamesOfMySeries.append(namesofMySeries[idx])
        flat_curves += flat_curve 
    return newMySeries, newNamesOfMySeries, flat_curves


def interpolate_series(series:"np.array",
                       interpolation_pts:"int"=200,
                       interpolation_kind:"str"="linear")->"np.array":
    '''
    Function interpolates time series with timestamped
    values using scipy interp1d. Input series should
    be 2D where the 1st dimension is the length of the
    time series and the 2nd dimension is a time and value.
    
    ex: Series is of shape (60, 2). This indicates that
    the series is 60 points long where each point is a
    time and value.
    
    Input:
    series = np.array([[0, 0.1], [1, 0.2], ... [59, 6.0] ])
    series.shape = (60, 2)
    
    Output: Interpolating to 200 points using linear interp.
    newseries = np.array([[0, 0.1], [1, 0.111], ... [200, 6.0] ])
    series.shape = (200, 2)
    '''
    x = series[:, 0]
    y = series[:, 1]
    f = interpolate.interp1d(x, y, kind=interpolation_kind)
    newx = np.linspace(x.min(), x.max(), interpolation_pts)
    newy = f(newx)
    newseries = np.column_stack((newx, newy))
    return newseries

def interpolate_series_list(series_list: "list[pd.DataFrame]",
                            **interpolate_series_kwargs)->"list[pd.DataFrame]":
    '''
    Function Takes list of time series dataframes and returns 
    a list of interpolated time series dataframes. This is used
    in the last stage of Msmc_clustering.read_file().
    '''
    interpolated_series_list = []
    for series in series_list:
        tmp_series = series.to_numpy()
        interpolated_series = interpolate_series(tmp_series, **interpolate_series_kwargs)
        df = pd.DataFrame(interpolated_series, columns=series.columns)
        interpolated_series_list.append(df)
    return interpolated_series_list
        

class Msmc_clustering():
    '''
    readfile_kwargs are fed to the readfile method which then feeds 
    readfile_kwargs to pd.read_csv! Make sure to specify sep parameter!
    '''
    def __init__(self,
                 directory,
                 mu=1,
                 generation_time_path=None,
                 to_omit=[],
                 exclude_subdirs=[],
                 manual_cluster_count=7,
                 algo="kmeans",
                 omit_front_prior=0,
                 omit_back_prior=0,
                 time_window=False,
                 time_field="left_time_boundary",
                 value_field="lambda",
                 interpolation_pts=200,
                 interpolation_kind='linear',
                 use_interpolation=False,
                 use_friendly_note=True,
                 use_real_time_and_c_rate_transform=False,
                 use_value_normalization=True,
                 use_time_log10_scaling=False,
                 use_plotting_on_log10_scale=False,  # Don't really need to touch unless you want to use Msmc_clustering in ipynb
                 data_file_descriptor=None,
                 **readfile_kwargs):
        
        if use_friendly_note:
            print(f"FRIENDLY NOTE 1 if getting err while reading data for Msmc_clustering: \
                    \nBy default, Msmc_clustering reads data from directory: <> using pd.read_csv with default params. \
                    \nMAKE SURE TO SPECIFY YOUR DESIRED sep. sep is read in through **readfile_kwargs and is a param \
                    \nin pd.read_csv. pd.read_csv has sep=\',\' by default. If the data held in <> are .tsv \
                    \nfiles, <sep = \'\t\'>")
            
        # #### ATTRIBUTES:
        # Bools
        self.use_value_normalization = use_value_normalization  # Either normalize lambda or make lambda on log10 scale
        self.use_real_time_and_c_rate_transform = use_real_time_and_c_rate_transform
        self.use_plotting_on_log10_scale = use_plotting_on_log10_scale
        self.use_time_log10_scaling = use_time_log10_scaling 
        self.use_interpolation = use_interpolation
        # time_window should not be used if you want to cluster, this is meant for reading in data
        # time_window will almost always cause filter_data() to fail because of how non-uniform
        # the dataset will become.
        # TIME WINDOW SHOULD HAVE REAL YEAR VALUES, NOT LOG10 TRANSFORMED VALUES
        # time_window should be a list/tuple containing a lower and upper bound on the time window you desire
        # Data settings
        self.time_window = time_window  # Enter a list of len=2, where first item is lower bound and second item is upper bound on desired time window of data (time is likely on log10 scale depending on settings)
        if not self.time_window:
            self.lowerbound = False
            self.upperbound = False
        elif len(self.time_window) == 2:
            self.lowerbound, self.upperbound = self.time_window
        self.interpolation_pts=interpolation_pts
        self.interpolation_kind=interpolation_kind
        self.time_field = time_field
        self.value_field = value_field
        self.mu = mu  # Possible that files use different mutation rates
        self.to_omit = to_omit  # List of file names to omit
        self.data_file_descriptor = data_file_descriptor # This is meant for the read_file method... Pretty f'n screwy since its not an actual arg
        self.omit_front_prior = omit_front_prior # Omit time points before saving data to mySeries
        self.omit_back_prior = omit_back_prior
        print(self.data_file_descriptor, "type: ", type(self.data_file_descriptor))
        
        print(f"\nREADING DATA")
        print(f"read_file summary:")
        print(f"omit_front_prior={self.omit_front_prior}\nomit_back_prior={self.omit_back_prior}\n")
        self.gen_time_dict = self.read_gen_times(generation_time_path)  # keys are latin names delim by "_" instead of " "
        tmp_data = self.read_file(directory,
                                  use_real_time_and_c_rate_transform,
                                  exclude_subdirs,
                                  **readfile_kwargs)
        self.mySeries, self.namesofMySeries, self.series_lengths, self.series_lengths_lists = tmp_data
        # self.log10_scale_time_and_normalize_values()
        if use_friendly_note:
            print(f"\nFRIENDLY NOTE 2 if getting nothing in mySeries after reading data with Msmc_clustering: \
                    \nDepending on the time_window you select, there may be no points in your input data which \
                    \nfit into the time_window. Make sure time window is in real time (this means you need to set \
                    \nuse_real_time_and_c_rate_transform = True)! Also, if using time_window, you probably won't be \
                    \nable to cluster data just yet. You would want to dump windowed data from the mySeries attribute \
                    \ninto a directory, interpolate the data in that directory so all files are of a uniform length, \
                    \nand read real time processed data which has been interpolated with a new Msmc_clustering instance \
                    \nwith use_real_time_and_c_rate_transform = False and maybe use_value_normalization=False and \
                    \nuse_time_log10_scaling=False if data was processed with Msmc_clustering where use_value_normalization=True \
                    \nand use_time_log10_scaling=True \
                    \nex: time window for pleistocene to last glacial period might be something like: time_window = [1.17E4, 2.58E6]\n")
        self.lenMySeries = len(self.mySeries)
        self.name2series = dict()  # Dict important for mapping names to series, label,
        self.dtw_labels = None  # Will be list of labels for how curves are clustered
        self.exhibitted_cluster_count = None
        self.clusterTable = None
        self.latin_names = None
        self.km = None
        self.label2barycenter = None  # Dict that is created when runing self.find_cluster_barycenters() after clustering
        self.flat_curves = 0  # Upon normalization, we shall find how many time series curves only have 1 unique y-value (flat)
        # ## CLUSTERING SETTINGS
        self.manual_cluster_count = manual_cluster_count
        self.algo = algo
        # ## INIT SOME STUFF
        if self.time_window == False:
            self.filter_data()
        else:
            print("we got a window")
            print('before:', len(self.mySeries))
            self.filter_data()
            print('after:', len(self.mySeries))
        self.name2series = {name: self.mySeries[idx] for idx, name in enumerate(self.namesofMySeries)}
        '''
        self.name2series:
        Important for mapping names to series. Particularly useful for grabbing
        data given a name, or given a list of names corresponding to a clustering
        group.

        EX:
        list_of_samples_of_label_1 = cluster_rt_norm_lenient.cluster_from_label(1)["Sample"]
        data_for_each_sample_of_label_1 = [self.name2series[name] for name in list_of_samples_of_label_1]
        '''
        if self.use_real_time_and_c_rate_transform:
            if self.use_value_normalization:
                if self.use_plotting_on_log10_scale:
                    self.suptitle = 'Effective Population Size Time Series'
                    self.xlabel = "Real Time in Years (log10)"
                    self.ylabel = "Effective Population Size (Normalized to [0, 1])"
                else:
                    self.suptitle = 'Effective Population Size Time Series Curves'
                    self.xlabel = "Real Time in Years"
                    self.ylabel = "Effective Population Size (Normalized to [0, 1])"   
            else:
                if self.use_plotting_on_log10_scale:
                    self.suptitle = 'Effective Population Size Time Series Curves'
                    self.xlabel = "Real Time in Years (log10)"
                    self.ylabel = "Effective Population Size (1E4)"
                else:
                    self.suptitle = 'Effective Population Size Time Series Curves'
                    self.xlabel = "Real Time in Years "
                    self.ylabel = "Effective Population Size (1E4)"   
        else:
            self.suptitle = 'Coalescence Rate Time Series Curves'
            self.xlabel = f"Scaled Time ({self.time_field})"
            self.ylabel = f"Scaled Coalescence Rate ({self.value_field})"

    # #### METHODS
    def read_file(self, directory, use_real_time_and_c_rate_transform, exclude_subdirs=[], **read_csv_kwargs):
        '''
        Data requirements:
        - Must be delimited by tabs, commas, etc.
        - Must have at least 2 columns (x and y or time point and value)
        - 1st row must contain names/labels for each column
        - All files should follow the same format
            - Same file descriptors (data_file_descriptores like .tsv, .csv, .txt, etc.)
            - Same headers/column names
        
        POTENTIAL ISSUE: When given files with a number of fields that is different from 4 (Specifically the MSMC fields)
        or 2 (plain old x, y fields), we will get an issue (see hardcoding of 'right_time_boundary' and such on df's).
        Possible solution is to allow function to take in arguments specifying the possible fields and the ones which we
        want to keep. Kinda like a pandas dataframe (is this a hint???).
        
        Assuming that given directory contains subdirs full of separate MSMC curves,
        go through each subdir in directory and read each file in subdir as a pd.df.

        each subdir may have a specific mu or generation time

        Input:
        1.) String of directory to read in. Should end with "/". 
        2.) use_real_time_and_c_rate_transform: If True, converts data to real time and lambda to Ne
        3.) pd.read_csv kwargs that aren't the filename
        4.) data_file_descriptor: file descriptor used. ex: .txt, .csv, .tsv

        Outputs:
        1.) List of dataframes (series)
        2.) List of the names corresponding to each df 
        3.) Set of unique series lengths 
        4.) List of series lengths
        '''
        mySeries = []
        namesofMySeries = []
        # print(self.to_omit)
        for subdir in os.listdir(directory):  # There is an assumption that each subdir has its own mu since each subdir has corresponded to a single tax-class
            if subdir not in exclude_subdirs: # If specified file data_file_descriptor matches, assume file
                filename = subdir  # Renamed subdir to filename for clarity
                if filename not in self.to_omit: # If data isn't to be omitted
                    df = pd.read_csv(directory + "/" + filename, usecols=[self.time_field, self.value_field], **read_csv_kwargs)
                    df = df.iloc[self.omit_front_prior:len(df)-self.omit_back_prior]  # Perform omission of points prior to saving in self.mySeries
                    
                    if use_real_time_and_c_rate_transform:  # If real time curves are desired, transform current df (Only use for MSMC/PSMC formatted data)
                        # Convert scaled time to real time
                        df[self.time_field] = df[self.time_field] / self.mu  # Convert scaled time to generations
                        for key in self.gen_time_dict.keys():  # Step can be improved if keys list is sorted
                            if key in filename:
                                generation_time = self.gen_time_dict[key]
                                df[self.time_field] = df[self.time_field] * generation_time # Convert generations to real time
                        # print(df)
                        df[self.value_field] = 1 / df[self.value_field]  # Take inverse of coalescence rate
                        df[self.value_field] = df[self.value_field] / (2 * self.mu)
                    mySeries.append(df)
                    rev_cut = filename[::-1].index('.')
                    name = filename[::-1][rev_cut:][::-1]
                    namesofMySeries.append(name)
        # HERE MIGHT BE GOOD TO WINDOW OFF DATA
        if self.time_window:
            # print('len of my series before windowing:', len(mySeries))
            mySeries, namesofMySeries = windowMySeries(mySeries=mySeries,
                                                       namesofMySeries=namesofMySeries,
                                                       by=self.time_field,
                                                       upperbound=self.upperbound,
                                                       lowerbound=self.lowerbound)
            print('len of my series after windowing:', len(mySeries))
        if self.use_time_log10_scaling or self.use_value_normalization: # Performed after windowing so we can window in real time if performing transforms
            mySeries, namesofMySeries, self.flat_curves = log10_scale_time_and_normalize_values(mySeries= mySeries,
                                                                                                namesofMySeries=namesofMySeries,
                                                                                                use_time_log10_scaling=self.use_time_log10_scaling,
                                                                                                use_value_normalization=self.use_value_normalization,
                                                                                                time_field=self.time_field,
                                                                                                value_field=self.value_field)

        if self.use_interpolation:
            mySeries = interpolate_series_list(mySeries,
                                               interpolation_pts=self.interpolation_pts,
                                               interpolation_kind=self.interpolation_kind)

        series_lengths = {len(series) for series in mySeries}  # Compile unique Series lengths
        series_lengths_list = [len(series) for series in mySeries]  # Compile unique Series length
        # print(series_lengths)
        # print(series_lengths_list)
        
            
        return mySeries, namesofMySeries, series_lengths, series_lengths_list

    def read_gen_times(self, directory: "str") -> "dict":
        '''
        Note: Pretty old method, hasn't really been proofed like most other
        methods, but it works.
        
        Makes a dict where the scientific names of species are the keys and their
        corresponding generation lengths are the values. This method takes a
        path to a directory containing files (tab separated) where each row 
        contains 1.) scientific name and 2.) generation length.
        
        Possibly better to read generation times data after reading in MSMC curves
        whenever number taxa generation times > ts curves available

        For each file in directory, reads in a two column txt file and returns a dict where keys are 
        sample-taxa name and values are generation times (years/generation) used
        for generating MSMC curves for each sample-taxa. Strong assumption that
        1st col is for sample-taxa names and 2nd col is for generation times

        Dict use:
        If file name has a dict key within it, use that key's value as a generation 
        time
        '''
        if directory:
            gen_time_dict = dict()
            genLenFile = os.listdir(directory)
            for glf in genLenFile:
                fileLen = 0
                duplicateEntries = 0
                with open(directory + glf, "r") as myfile:
                    next(myfile) # Assuming that first line in generation times file are labels
                    for idx, line in enumerate(myfile):
                        pair = line.split("\t")
                        # print(pair)
                        key = pair[0] # taxa name
                        val = float(pair[1]) # generation time
                        gen_time_dict[key] = val
                        fileLen += 1
                        if fileLen - len(list(gen_time_dict.keys())) > duplicateEntries:
                            print(f"Duplicate gen time entry for taxa: {key} at line {idx}")
                            duplicateEntries += 1
                    print(f"fileLen: {fileLen}\nduplicate entries: {duplicateEntries}")
            return gen_time_dict
        else:
            return None

    def filter_data(self):
        '''
        filters out time points == 0 (logically we only have 1 value per time
        point so the most points which can be 0 should be 1)
        
        First part takes in a group of series and scales their time ranges to
        the series with the oldest date.

        Inputs:
        1.) Set of unique series lengths
        2.) List of series lengths 
        3.) Option for scaling 
        '''
        # Record sizes and lengths of series
        # if scaled_to_real: # Scale data to real time and eff. pop. sizes

        # Only use series' of the longest known length
        if len(self.series_lengths) > 0:  
            max_series_len = max(self.series_lengths)
            newMySeries = []
            newNamesOfMySeries = []
            for idx, series in enumerate(self.mySeries):
                if len(series) == max_series_len or self.time_window: # "or self.time_window" is included in condition for enabling the reading of time window'd data
                    # print("filter_data")
                    # print(self.time_field)
                    # print(series)
                    # if series[self.time_field].iloc[0] == 0:  # If 1st time entry is 0
                    #     # print(f"len of trimmed df: {len(series.iloc[1:])}")
                    #     newMySeries.append(series.iloc[1:])  # Clip off 1st entry to avoid -inf err when scaling to log10 scale in log10_scale_time_and_normalize_values()
                    #     # Entry at 0th and 1st idx are identical for lambda so no meaningful info should be lost
                    # else:
                    newMySeries.append(series)
                    newNamesOfMySeries.append(self.namesofMySeries[idx])
            self.mySeries = newMySeries
            self.namesofMySeries = newNamesOfMySeries

    def plot_series(self, num_to_plot=None, cols=5, fs_x=50, fs_y=25, **step_kwargs):
        '''
        Plots curves for mySeries dfs as they are in current object. By default
        mySeries dfs may only be in terms of Scaled time and Coalescence Rate

        Inputs:
        1.) Int of plots to make
        2.) Int of columns for figure
        '''
        if not num_to_plot:  # If no number of plots is specified
            num_to_plot = self.lenMySeries  # Number of plots to plot is set to the number of files read in
        # rows = (num_to_plot//cols) + 1 # Find number of rows for figure
        if num_to_plot % cols > 0:
            rows = (num_to_plot // cols) + 1  # Find number of rows for figure
        else:
            rows = (num_to_plot // cols)
        fig, axs = plt.subplots(rows, cols, figsize=(fs_x, fs_y))  # MPL subplot
        fig.suptitle(self.suptitle)  # Figure super title

        for i in range(rows):
            for j in range(cols):
                if i*cols+j+1 > len(self.mySeries):  # pass the others that we can't fill
                    continue
                curr = self.mySeries[i*cols+j]
                x_list = curr[self.time_field].to_numpy()
                y_list = curr[self.value_field].to_numpy()
                axs[i, j].step(x_list, y_list, 'g-', where="pre", **step_kwargs)
                axs[i, j].set_title(self.namesofMySeries[i*cols+j])
                axs[i, j].set_xlabel(self.xlabel)  # time
                axs[i, j].set_ylabel(self.ylabel)  # size
        fig.patch.set_facecolor('white')  # Changes background to white
        plt.show()

    def compute_cluster_barycenter(self, label: "int", iter: "int" = 5, gamma=None, **barycenter_averaging_kwargs):
        '''
        Returns barycenter of a computed cluster as two lists: x, y
        '''

        data = []
        names = self.cluster_from_label(label)["Sample"].tolist()
        for name in names:
            data.append(self.name2series[name].to_numpy())
        if gamma is not None and not gamma == 0.0:
            barycenter = softdtw_barycenter(data, max_iter=5, gamma=gamma, **barycenter_averaging_kwargs)
        else:
            # print('regular dtw?')
            barycenter = dtw_barycenter_averaging(data, max_iter=5, **barycenter_averaging_kwargs)
        # print(barycenter)
        x, y = zip(*barycenter)
        return x, y

    def find_cluster_barycenters(self, iter:"int"=5, gamma=None, **barycenter_averaging_kwargs):
        '''
        Acts with self.compute_cluster_barycenter()
        '''
        label2barycenter = dict()
        for label in set(self.dtw_labels): # self.dtw_labels computed after clustering, only using the set of labels though, so might just want to use the range(clusternumber)
            label2barycenter[label] = self.compute_cluster_barycenter(label, iter, gamma, **barycenter_averaging_kwargs)
        return label2barycenter

    def cluster_curves(self,
                       manual_eps=False,
                       random_state=205,
                       omit_front=0,
                       omit_back=0, cols=3, fs_x=25, fs_y=50, save_to=None,
                       metric="dtw",
                       save_name="cluster-related-figures/curve-clusters.png",
                       omit_time_feature=False, metric_params={"gamma": None},
                       iter=100, plot_everything=True, min_samples=5,
                       only_names=False,
                       plot_barycenters=True,
                       **kwargs):
        '''
        If using Kmeans (main algo used), all data entries (items in mySeries)
        must be of the same length. Training won't work otherwise. Training
        on data where time_window=False should be fine for training if you follow
        the assumptions for Msmc_clustering data. If you entered a time_window,
        clustering will likely end in failure. To cluster windowed data, interpolate
        the data which you just windowed to and equal number of data points like 
        50, 100, 200, etc.
        
        DBSCAN notes:

        eps: The maximum distance between two samples for one to be considered 
        as in the neighborhood of the other. This is not a maximum bound on the 
        distances of points within a cluster. This is the most important DBSCAN 
        parameter to choose appropriately for your data set and distance 
        function.

        ex: Consider 2 seqs compared using DTW
        x = [7.0, 7, 7, 7, 7, 6.7, 7.7, 7, 7, 7, 7, 7.6, 7, 6.7, 7, 7, 7, 7, 7, 7.1, 7, 7, 7, 7, 7]
        y = [10., 7, 7, 7, 7, 6.7, 7.7, 7, 7, 7, 7, 7.6, 7, 6.7, 7, 7, 7, 7, 7, 7.1, 7, 7, 7, 7, 7]
        dtw_dist(x, y) is 3.0. Notice that 7 and 10 in position 0 are the only 
        points differing (differ by 3). Dist is as straight forward as that

        In our case with N number of ts of shape (1, 63, 2) where y vals are 
        [0, 1], I probably should make eps < 63. Maybe something like the 
        sqrt(N) lol

        - only_names: List of sample names to cluster
        '''
        if manual_eps:
            eps = manual_eps
        else:
            eps = np.sqrt(self.lenMySeries)

        # Form a list of dfs to cluster
        # By default do this b/c tslearn has a divide by 0 err o.w.
        if metric == "softdtw" and (metric_params["gamma"] == 0.0 or metric_params["gamma"] is None):
            metric = 'dtw'
        if only_names:
            # List of dfs with only the desired 2 columns
            cleanSeries_with_df = [series
                                   for series in self.mySeries]
            # selects series to keep from given only_names list
            cleanSeries_with_df = [cleanSeries_with_df[i] for i in
                                   range(len(cleanSeries_with_df))
                                   if self.namesofMySeries[i] in only_names]                    
            # print(cleanSeries_with_df)
        else:
            # List of dfs with only the desired 2 columns
            cleanSeries_with_df = [series
                                   for series in self.mySeries]
        print(f"cleanSeries_with_df: {len(cleanSeries_with_df)},")
        print(f"mySeries: {len(self.mySeries)}")
        print(f"len of series in mySeries: {len(self.mySeries[0])}")
        # Omit certain number of datapoints from front (end time) and back (beginning time) of ts curve
        cleanSeries_with_df = [series[omit_front: max(self.series_lengths) - omit_back] for series in cleanSeries_with_df] 
        cleanSeries = np.array(cleanSeries_with_df)
        # After checking how clusters are formed when omitting time data, worse clusters will form
        if omit_time_feature:  # Enable only if you want to exclude time interval info from clustering
            cleanSeries = cleanSeries[:, :, 1] 
        print(f"Clustering {len(cleanSeries[0])}/{max(self.series_lengths)}")
        print(f"Omitting {omit_front} points from front and {omit_back} from back")
        # Cluster using dtw 
        print(self.algo, metric)
        print(f"type of cleanSeries {type(cleanSeries)}")
        print(f"Number of flat time series curves: {self.flat_curves}")
        self.cleanSeries = cleanSeries
        # Choose distance metric for clustering
        # # cleanSeriesXY: array-like of shape=(n_ts, sz, d) ts data.
        cleanSeriesXY = cleanSeries.reshape(cleanSeries.shape[0], -1,)
        # km.fit_predict is fit on a 2 column df if using dists
        if metric == "softdtw":
            gamma = metric_params["gamma"]
            dists = pairwise_distances(cleanSeriesXY,
                                       metric=softdtw_distance,
                                       gamma=gamma)
        elif metric == "dtw":
            # print()
            # print(cleanSeriesXY)
            # print(len(cleanSeries))
            # print(cleanSeries[0].shape)
            dists = pairwise_distances(cleanSeriesXY,
                                       metric=dtw_distance)
        else:
            print("Using stock pairwise distance metric")
            dists = pairwise_distances(cleanSeriesXY,
                                       metric=metric)
        print(f"Computed dists of shape: {dists.shape}")
        # Choose algorithm for clustering
        if self.algo == "dbscan":  # sklean algo   
            self.km = DBSCAN(metric="precomputed",
                             metric_params=metric_params,
                             eps=eps,
                             min_samples=min_samples)
            self.dtw_labels = self.km.fit_predict(dists)
        # elif self.algo == "hierarchicalComplete":
        elif self.algo == "FeatureAgglomeration":  # sklean algo
            linkage_matrix = hierarchical_clustering(dists)
            self.km = FeatureAgglomeration(metric="precomputed",
                                           n_clusters=self.manual_cluster_count)
            self.dtw_labels = self.km.fit_predict(linkage_matrix)
        elif self.algo == "sk_kmeans":  # sklearn algo
            self.km = KMeans(n_clusters=self.manual_cluster_count,
                             random_state=random_state,
                             max_iter=iter)
            self.dtw_labels = self.km.fit_predict(dists)
        elif self.algo == "kmeans":  # tslearn algo
            if metric == "softdtw":
                self.km = TimeSeriesKMeans(n_clusters=self.manual_cluster_count,
                                           verbose=False,
                                           metric="softdtw",
                                           metric_params=metric_params,
                                           dtw_inertia=True,
                                           random_state=random_state,
                                           max_iter=iter,
                                           max_iter_barycenter=iter)
                self.dtw_labels = self.km.fit_predict(cleanSeries)
            else:
                self.km = TimeSeriesKMeans(n_clusters=self.manual_cluster_count,
                                           verbose=False,
                                           metric=metric,
                                           dtw_inertia=True,
                                           random_state=random_state,
                                           max_iter=iter,
                                           max_iter_barycenter=iter)
                # print(cleanSeries)
                self.dtw_labels = self.km.fit_predict(cleanSeries)
        elif self.algo == "kshape":  # tslearn algo
            print(f"manual cluster number: {self.manual_cluster_count}")
            self.km = KShape(n_clusters=self.manual_cluster_count,
                             max_iter=iter,
                             random_state=random_state,
                             init="random")
            self.dtw_labels = self.km.fit_predict(cleanSeries)
        print(type(self.km))
        # Make table mapping names to labels
        self.clusterTable = {self.namesofMySeries[i] : self.dtw_labels[i] for i in range(len(self.dtw_labels))}
        print(f"sample len: {len(self.namesofMySeries)}")
        print(f"label len: {len(self.dtw_labels)}")
        if only_names:
            data = {"Sample": only_names, "Labels": self.dtw_labels}
        else:
            data = {"Sample": self.namesofMySeries, "Labels": self.dtw_labels}
        self.clusterTable = pd.DataFrame.from_dict(data, orient="columns")
        self.add_latin_to_cluster()
        # for idx, name in enumerate(self.namesofMySeries):
        #     self.name2series[name] = self.mySeries[idx]
        # Compute some cluster barycenters
        gamma = metric_params["gamma"]
        self.label2barycenter = self.find_cluster_barycenters(iter, gamma)
        # Plots curves within their assigned clusters
        if plot_everything:
            self.exhibitted_cluster_count = len(set(self.dtw_labels))
            self.plot_curve_clusters(cleanSeries_with_df, cols, fs_x, fs_y,
                                     save_to=save_to,
                                     metric_params=metric_params,
                                     save_name=save_name,
                                     plot_barycenters=plot_barycenters,
                                     **kwargs)
            # Plots distribution of curves among the clusters created with dtw
            self.plot_cluster_distr(save_to=save_to,
                                    save_name="distr_" + save_name)
            print("DID I CLOSE THE PLOT??? PLEASE TELL ME I DID!")
            plt.close()

    def to_training_data(self):
        '''
        Method returns time series as an 3 dimensional tensor. 1st dimension
        corresponds to samples/entries. 2nd dimension corresponds to the length of
        an entry. 3rd dimension corresponds to actual time point and value in a
        segment of the time series.

        Note: If you want the names corresponding to each sample in the output,
        access the namesofmySeries attribute for a list of names corresponding
        to each sample.

        ex: to_training_data returns data of the shape (100, 60, 2). There are
        100 samples in this dataset. Each sample has a time series of length
        60. Each "point" in the time series has 2 fields which are usually a
        specific timestamp and a value. In terms of the MSMC the time stamp is
        time in the past and the value is NE.
        '''
        X = np.array([i.to_numpy() for i in self.mySeries])
        return X

    def plot_curve_clusters(self,
                            cleanSeries,
                            cols=3,
                            fs_x=50,
                            fs_y=25,
                            save_to=None,
                            iter=5,
                            metric_params={"gamma": None},
                            save_name="curve-clusters.png",
                            plot_barycenters=True,
                            plot_iceages=False,
                            **kwargs):
        '''
        This function is admittidly overcomplicated. I have another somewhere
        that is much simpler. It is in the ANOVA-birds ipynb
        
        Plots curves within their clusters assigned from K means using DTW

        Dark Grey region is Last Glacial Period
        Light Grey region is Pleistocene
        '''
        # Necessary for plotting figure subplots (square shape), nothing else
        # plot_count = math.ceil(math.sqrt(self.exhibitted_cluster_count))
        num_to_plot = self.exhibitted_cluster_count  # Set no. plots to num clusters
        print(f"num to plot : {num_to_plot}")
        if num_to_plot % cols == 0:
            rows = (num_to_plot//cols)
        else:
            rows = (num_to_plot//cols) + 1
        print(f"curve cluster plot shape: ({rows}, {cols})")
        fig, axs = plt.subplots(rows, cols, figsize=(fs_x, fs_y))
        if self.algo == "kmeans":
            if metric_params["gamma"] is None:
                fig.suptitle(f'DTW Clusters of {self.suptitle}\n \
                             K = {self.exhibitted_cluster_count}')
            else:
                fig.suptitle(f'soft-DTW Clusters of {self.suptitle}\n{metric_params}, K = {self.exhibitted_cluster_count}')
        else:
            fig.suptitle(f'{self.algo} Clusters of {self.suptitle}')
        plt.rcParams["figure.facecolor"] = 'white'

        row_i=0
        column_j=0
        # Plot curves with their cluster according to their label
        for idx_l, label in enumerate(set(self.dtw_labels)): # For each unique/possible label
            x_cluster = []
            y_cluster = []
            for i in range(len(self.dtw_labels)): # For each curve's label
                    if self.dtw_labels[i]==label: # match it to the current unique/possible label focused on
                        if self.use_plotting_on_log10_scale and not self.use_time_log10_scaling:
                            x = log10_with_zero(cleanSeries[i][self.time_field].to_numpy()) # Index mySeries for df
                        else:
                            x = cleanSeries[i][self.time_field].to_numpy()
                        y = cleanSeries[i][self.value_field].to_numpy()


                        # Curve color assignment
                        reg_curve_color = "gray" 

                        if rows == 1: # If axs only takes 1D indices
                            axs[column_j].step(x, y, "+-", c=reg_curve_color, alpha=0.4)
                            
                        else:
                            axs[row_i, column_j].step(x, y, "+-", c=reg_curve_color,alpha=0.4)
                        x_cluster.append(x)
                        y_cluster.append(y)
            if len(x_cluster) > 0:
                # X = dtw_barycenter_averaging(x_cluster, y_cluster)
                x_avg = np.average(x_cluster, axis=0)
                y_avg = np.average(y_cluster, axis=0)
                y_sd = np.std(y_cluster, axis=0)

                if rows == 1: # If axs only takes 1D indices

                    axs[column_j].step(x_avg ,y_avg, "+--",    c="magenta", alpha=0.4, label="Arithmetic mean") # Arithemtic avg of curves plotted in magenta
                    axs[column_j].step(x_avg ,y_avg-y_sd, "+--", c="blue", alpha=0.4, label="Arithmetic SD Lower") 
                    axs[column_j].step(x_avg ,y_avg+y_sd, "+--", c="blue", alpha=0.4, label="Arithmetic SD Upper") 
                    # Plot cluster centroid
                    # center_series = self.km.cluster_centers_[label]
                    # x_center = [i[0] for i in center_series]
                    # y_center = [i[1] for i in center_series]
                    # axs[column_j].step(x_center, y_center, "+--", c="orange", alpha=0.6, label="Cluster Centroid")
                    # Plot cluster barycenter
                    if plot_barycenters:
                        x_barycenter, y_barycenter = self.label2barycenter[label]
                        axs[column_j].step(x_barycenter, y_barycenter, "+-", c="red", alpha=1, label="Cluster Barycenter")

                    # axs[column_j].step(xs, ys, "+-", label="DTW Barycentric avg") # DBA line which DTW uses as centroids for clustering
                    axs[column_j].set_title("Cluster "+ str(column_j + (row_i*cols)) )
                    axs[column_j].set_xlabel(self.xlabel)
                    axs[column_j].set_ylabel(self.ylabel)
                    if self.use_real_time_and_c_rate_transform and plot_iceages:  # Here is where to edit for differentiating curves by color
                        axs[column_j].axvspan(4, 5, alpha=0.5, color='grey')
                        axs[column_j].axvspan(5, 6.3, alpha=0.25, color='grey')
                        # axs[column_j].set_xlim(4, 7.5)
                else: 
                    axs[row_i, column_j].step(x_avg ,y_avg, "+--",    c="magenta", alpha=0.4, label="Arithmetic mean") # Arithemtic avg of curves plotted in magenta
                    axs[row_i, column_j].step(x_avg ,y_avg-y_sd, "+--", c="blue", alpha=0.4, label="Arithmetic SD Lower") 
                    axs[row_i, column_j].step(x_avg ,y_avg+y_sd, "+--", c="blue", alpha=0.4, label="Arithmetic SD Upper") 
                    # Plot cluster center
                    # center_series = self.km.cluster_centers_[label]
                    # x_center = [i[0] for i in center_series]
                    # y_center = [i[1] for i in center_series]
                    # axs[row_i, column_j].step(x_center, y_center, "+--", c="orange", alpha=0.6, label="Cluster Centroid")
                    # Plot cluster barycenter
                    if plot_barycenters:
                        x_barycenter, y_barycenter = self.label2barycenter[label]
                        axs[row_i, column_j].step(x_barycenter, y_barycenter, "+-", c="red", alpha=1, label="Cluster Barycenter")

                    # axs[row_i, column_j].step(xs, ys, "+-", c="red", label="DTW Barycentric avg") # DBA line which DTW uses as centroids for clustering
                    axs[row_i, column_j].set_title("Cluster "+ str(column_j + (row_i*cols)) )
                    axs[row_i, column_j].set_xlabel(self.xlabel)
                    axs[row_i, column_j].set_ylabel(self.ylabel)
                    if self.use_real_time_and_c_rate_transform and plot_iceages:
                        axs[row_i, column_j].axvspan(4, 5, alpha=0.5, color='grey')
                        axs[row_i, column_j].axvspan(5, 6.3, alpha=0.25, color='grey')
                        # axs[row_i, column_j].set_xlim(4, 7.5)
            
            column_j+=1 # Increment subplot column
            if column_j == cols:
                row_i+=1
                column_j=0
        if save_to:
            plt.savefig(save_to + save_name, dpi = 300)
        plt.show()

    def plot_cluster_distr(self, save_to=None, save_name="curve-clusters.png"):
        '''
        Plots histogram to illustrate distribution of curves among the clusters 
        created with dtw
        '''
        cluster_c = [len(self.dtw_labels[self.dtw_labels==i]) for i in range(self.exhibitted_cluster_count)]
        cluster_n = ["Cluster "+str(i) for i in range(self.exhibitted_cluster_count)]
        plt.figure(figsize=(15,5))
        plt.title(f"Cluster Distribution for {self.algo}")
        plt.bar(cluster_n,cluster_c)
        if save_to:
            plt.savefig(save_to + save_name, dpi = 100)
        plt.show()


    def plot_curve(self, name=None, df = None, dir = None, dups=0, stretch=0, err=0, thresh_min=None, thresh_max=None, winStart=None, winEnd=None, fs_x=10, fs_y=10, xlim_start=None, xlim_end=None, ylim_start=None, ylim_end=None,  save_to = None, additional_info=None, use_plotting_on_log10_scale=True):
        '''
        Plots series curve given name, df, or dir of series
        Option to save plot to a directory
        '''
        # plt.rcParams["figure.facecolor"] = 'white'
        plt.figure(figsize=(fs_x, fs_y))

        if xlim_start and xlim_end:
            plt.xlim(xlim_start, xlim_end)
        if ylim_start and ylim_end:
            plt.ylim(ylim_start, ylim_end)
        

        if name:
            idx = self.namesofMySeries.index(name)
            series = self.mySeries[idx]
        elif dir:
            df = pd.read_csv(dir, sep='\t')
            # While we are at it I just filtered the columns that we will be working on
            df.set_index("time_index",inplace=True)
            # set the date columns as index
            df.sort_index(inplace=True)
            name = dir[:dir.index(".txt")]
            series = df
        else:
            series = df

        if use_plotting_on_log10_scale and not self.use_time_log10_scaling:
            x = log10_with_zero(series[self.time_field].to_numpy()) # Index mySeries for df
            # if winStart and winEnd:
            #     winStart = log10_with_zero(winStart)
            #     winEnd = log10_with_zero(winEnd)
        else:
            x = series[self.time_field].to_numpy()
        y = series[self.value_field].to_numpy()
        plt.axvspan(4, 5, alpha=0.5, color='grey')
        plt.axvspan(5, 6.3, alpha=0.25, color='grey')
        plt.step(x, y, "+-", c="green")
        if thresh_min and thresh_max:
            plt.plot(x, [thresh_min]*len(x), "-", color="salmon")
            plt.plot(x, [thresh_max]*len(x), "--", color="red")
            
        if winStart and winEnd:
            plt.axvline(x = winStart, color="magenta", linestyle="-")
            plt.axvline(x = winEnd, color="purple", linestyle="--")
        if additional_info:
            title = f"{name} \n{dups*100:.2f}% of curve is flat.  Curve flatness threshold: >={stretch*100:.2f}%\nUsing {err*100}% error"
            for string in additional_info:
                title += string + "\n"
            plt.title(title)
        else:
            plt.title(f"{name} \n{dups*100:.2f}% of curve is flat.  Curve flatness threshold: >={stretch*100:.2f}%\nUsing {err*100}% error")
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        # plt.show()
        if save_to:
            plt.savefig(fname=f"{save_to}/{name}.png", dpi=300)
        plt.show()

    def list_from_cluster(self, num, omit=False):
        '''
        Return a list which either keeps or omits the names of the files of cluster
        corresponding to given {num } 
        '''
        if not omit:
            boolList = [True if i == num else False for i in self.clusterTable["Labels"].tolist()]
        else:
            boolList = [False if i == num else True for i in self.clusterTable["Labels"].tolist()]
        return self.clusterTable[boolList]["Sample"].tolist()

    def cluster_from_label(self, label: "int") -> "df":
        '''
        Retrieve a single cluster from self.clusterTable as a df given a label
        '''
        boolList = (self.clusterTable["Labels"] == label).tolist()
        return self.clusterTable.iloc[boolList]

    def series_from_name(self, name):
        idx = self.namesofMySeries.index(name)
        return self.mySeries[idx]

    def add_latin_to_cluster(self):
        '''
        Adds latin names column to cluster table in order to get some sort of relational
        identifier between it and the meta data table
        Table matcher only works if original MSMC curve files had their corresponding
        GCA accession ID in their name. However this loop only uses GCA accession ID to 
        throw it out, so GCA isn't really needed to accomplish the goal of confering 
        df_meta entries over to associated clusterTable entries. It's just the simplest 
        and most efficient way to do it right now, unless I can somehow add GCA accession
        ID to the
        '''
        latin_name_list = []
        for sample_name in self.clusterTable["Sample"]:
            if "_GC" in sample_name:
                if "_GCA" in sample_name:
                    cutoff_idx = sample_name.index("_GCA")  # Bad, but working way to separate
                elif "_GCF" in sample_name:
                    cutoff_idx = sample_name.index("_GCF")  # Bad, but working way to separate
                else:
                    print("I DON'T KNOW LATIN!")
                latin_name = sample_name[:cutoff_idx]
            else:
                latin_name = sample_name
            latin_name_list.append(latin_name)
        self.clusterTable["Latin name"] = pd.Series(latin_name_list)
        self.latin_names = self.clusterTable["Latin name"].tolist()
        self.clusterTable = self.clusterTable.set_index("Latin name")
        return
