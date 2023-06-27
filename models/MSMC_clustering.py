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
# from tslearn
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import dtw
from tslearn.metrics import soft_dtw
from tslearn.clustering import KShape
from tslearn.clustering import TimeSeriesKMeans


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
    # print(f'upperbound by {by}:',(df[by] < upperbound))
    # print(f'lowerbound by {by}:',(df[by] > lowerbound))
    index = (df[by] < upperbound) & (df[by] > lowerbound)
    # print(df)
    # print(index)
    return df.loc[index]


def windowMySeries(mySeries, **kwargs):
    '''
    seriesWindow wrapper function.
    
    Takes in a list of time series dataframes and returns a truncated version 
    of that list of time series dataframes. Truncation is performed with
    seriesWindow function which takes in dataframes, a field to truncate by,
    an upperbound on the field, and a lowerbound on the field.
    '''
    windowedSeries = []
    for df in mySeries:
        windowedSeries.append(seriesWindow(df, **kwargs))
    return windowedSeries


class Msmc_clustering():
    '''
    readfile_kwargs are fed to the readfile method which then feeds 
    readfile_kwargs to pd.read_csv! Make sure to specify sep parameter!
    '''
    def __init__(self,
                 directory,
                 friendly_note=True,
                 mu=None,
                 generation_time_path=None,
                 real_time=False,
                 normalize_lambda=True,
                 log_scale_time=False,
                 plot_on_log_scale=False,
                 uniform_ts_curve_domains=False,
                 to_omit=[],
                 exclude_subdirs=[],
                 manual_cluster_count=False,
                 algo="kmeans",
                 suffix='.txt',
                 omit_front_prior=0, 
                 omit_back_prior=0,
                 time_window=False,
                 index_field="time_index",
                 time_field="left_time_boundary",
                 value_field="lambda",
                 ignore_fields:"list<str>"=["right_time_boundary"],
                 **readfile_kwargs):
        if friendly_note:
            print(f"FRIENDLY NOTE if getting err while reading data for Msmc_clustering:\n \
By default, Msmc_clustering reads data from directory: {directory} using pd.read_csv with default params.\n \
MAKE SURE TO SPECIFY YOUR DESIRED sep. sep is read in through **readfile_kwargs and is a param\n \
in pd.read_csv. pd.read_csv has sep=\',\' by default. If the data held in {directory} are .tsv\n \
files, sep = \'\t\' \
\n")
        # #### ATTRIBUTES:
        # ## DATA SETTINGS
        
        # time_window should not be used if you want to cluster, this is meant for reading in data
        # time_window will almost always cause filter_data() to fail because of how non-uniform
        # the dataset will become.
        # TIME WINDOW SHOULD HAVE REAL YEAR VALUES, NOT LOG10 TRANSFORMED VALUES
        self.time_window = time_window # Enter a list of len=2, where first item is lower bound and second item is upper bound on desired time window of data (time is likely on log10 scale depending on settings)
        if self.time_window:
            assert len(self.time_window) == 2, "time_window should be a list/tuple containing a lower and upper bound on the time window you desire"
            self.lowerbound, self.upperbound = self.time_window
        else:
            self.lowerbound = False
            self.upperbound = False
        self.index_field = index_field
        self.time_field = time_field
        self.value_field = value_field
        self.ignore_fields = ignore_fields
        self.mu = mu  # Possible that files use different mutation rates
        self.subdir_class_dict = {  # Helps to map mu to dir of data
            "birds_part_1": "aves",
            "birds_part_2": "aves",
            "mammals_part_1": "mammals",
            "Archive": "mammals"
        }
        self.class_mu_dict = {  # Maps taxonomical class to mu; GOOD FOR CLUSTERING WITH BIRDS AND MAMMALS WHICH HAVE DIFF MU's.
            "aves": 1.4e-9,
            "mammals": 2.2e-9
        }
        self.subdir2file_dict = {subdir: [] for subdir in self.subdir_class_dict.keys()} # GOOD FOR CLUSTERING WITH BIRDS AND MAMMALS WHICH HAVE DIFF MU's.
        self.normalize_lambda = normalize_lambda  # Either normalize lambda or make lambda on log10 scale
        self.real_time = real_time
        self.gen_time_dict = self.read_gen_times(generation_time_path)  # keys are latin names delim by "_" instead of " "
        if self.real_time:
            self.plot_on_log_scale = True
            self.log_scale_time = True
        else:
            self.plot_on_log_scale = plot_on_log_scale
            self.log_scale_time = log_scale_time 
        self.to_omit = to_omit  # List of file names to omit
        
        self.suffix = suffix # This is meant for the read_file method... Pretty f'n screwy since its not an actual arg
        self.omit_front_prior = omit_front_prior # Omit time points before saving data to mySeries
        self.omit_back_prior = omit_back_prior
        print(self.suffix, "type: ", type(self.suffix))
        print(f"\nread_file summary:")
        print(f"omit_front_prior={self.omit_front_prior}\nomit_back_prior={self.omit_back_prior}\n")
        tmp_data = self.read_file(directory,
                                  real_time,
                                  exclude_subdirs,
                                  **readfile_kwargs)
        self.mySeries, self.namesofMySeries, self.series_lengths, self.series_lengths_lists = tmp_data
        self.lenMySeries = len(self.mySeries)
        self.name2series = dict()  # Dict important for mapping names to series, label,
        self.dtw_labels = None  # Will be list of labels for how curves are clustered
        self.cluster_count = None
        self.clusterTable = None
        self.latin_names = None
        self.km = None
        self.elbow_data = None
        self.clustering_data = None  # From cleanSeries arrays used for clustering 
        self.label2barycenter = None  # Dict that is created when runing self.find_cluster_barycenters() after clustering
        self.flat_curves = 0  # Upon normalization, we shall find how many time series curves only have 1 unique y-value (flat)
        # ## CLUSTERING SETTINGS
        self.manual_cluster_count = manual_cluster_count
        self.algo = algo
        # ## INIT SOME STUFF
        if self.time_window == False:
            self.filter_data(uniform_ts_curve_domains)
        else:
            print("we got a window")
            print('before:', len(self.mySeries))
            self.filter_data(uniform_ts_curve_domains=False)
            print('after:', len(self.mySeries))
        # print('before normalization')
        # print(self.mySeries[9]['lambda'])
        # print(self.mySeries[9]['lambda'].unique())
        self.normalize()
        # print('after normalization')
        # print(self.mySeries[9]['lambda'])
        self.seriesDict = {name: self.mySeries[idx] for idx, name in enumerate(self.namesofMySeries)}

        '''
        self.name2series:
        Important for mapping names to series. Particularly useful for grabbing
        data given a name, or given a list of names corresponding to a clustering
        group.

        EX:
        list_of_samples_of_label_1 = cluster_rt_norm_lenient.cluster_from_label(1)["Sample"]
        data_for_each_sample_of_label_1 = [self.name2series[name] for name in list_of_samples_of_label_1]
        '''
        if self.real_time:
            if self.normalize_lambda:
                if self.plot_on_log_scale:
                    self.suptitle = 'Effective Population Size Time Series'
                    self.xlabel = "Real Time in Years (log10)"
                    self.ylabel = "Effective Population Size (Normalized to [0, 1])"
                else:
                    self.suptitle = 'Effective Population Size Time Series Curves'
                    self.xlabel = "Real Time in Years "
                    self.ylabel = "Effective Population Size (Normalized to [0, 1])"   
            else:
                if self.plot_on_log_scale:
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
    def read_file(self, directory, real_time, exclude_subdirs=[], window=False, **read_csv_kwargs):
        '''
        POTENTIAL ISSUE: When given files with a number of fields that is different from 4 (Specifically the MSMC fields)
        or 2 (plain old x, y fields), we will get an issue (see hardcoding of 'right_time_boundary' and such on df's).
        Possible solution is to allow function to take in arguments specifying the possible fields and the ones which we
        want to keep. Kinda like a pandas dataframe (is this a hint???).
        
        Assuming that given directory contains subdirs full of separate MSMC curves,
        go through each subdir in directory and read each file in subdir as a pd.df.

        each subdir may have a specific mu or generation time

        Input:
        1.) String of directory to read in. Should end with "/". 
        2.) real_time: If True, converts data to real time and lambda to Ne
        3.) pd.read_csv kwargs that aren't the filename
        4.) suffix: file descriptor used. ex: .txt, .csv, .tsv

        Outputs:
        1.) List of dataframes (series)
        2.) List of the names corresponding to each df 
        3.) Set of unique series lengths 
        4.) List of series lengths
        '''
        suff = self.suffix
        mySeries = []
        namesofMySeries = []
        # print(self.to_omit)
        for subdir in os.listdir(directory):  # There is an assumption that each subdir has its own mu since each subdir has corresponded to a single tax-class
            # Dependence of my on assumption that files in a subdir are of the same class can be circumvented if I just have a mapping between filenames and tax-classes
            # print(subdir+"/")
            if subdir not in exclude_subdirs:
                if "." in subdir:  # Then we got a file!
                    # print("We got a file")
                    # print(namesofMySeries)
                    filename = subdir
                    # if filename.endswith(suff): # Check for correct suffix
                    if suff == filename[-len(suff):]:  # Check for correct suffix
                        if filename[:-len(suff)] not in self.to_omit:
                            # print(directory+subdir+"/"+filename)
                            df = pd.read_csv(directory + "/" + filename, **read_csv_kwargs)
                            
                            
                            # ISSUE CODE BELOW! Might just feed usecols to kwargs
                            if not self.time_field:  # We need to impute index and a fake right_time_boundary col
                                impromptu_index = list(range(df.shape[0]))
                                # print(impromptu_index)
                                df.index = impromptu_index
                            else:
                                df.set_index(keys=[self.index_field], inplace=True)
                                
                            df = df.iloc[self.omit_front_prior:len(df)-self.omit_back_prior]  # Perform omission of points prior to saving in self.mySeries
                            df.sort_index(inplace=True)
                            # and lastly, ordered the data according to our date index
                            # self.subdir2file_dict[subdir].append(filename)
                            if real_time:  # If real time curves are desired, transform current df (Only use for MSMC/PSMC formatted data)
                                # mu = self.class_mu_dict[self.subdir_class_dict[subdir]]  # GOOD FOR CLUSTERING WITH BIRDS AND MAMMALS WHICH HAVE DIFF MU's. Index into my convoluted ass dictionaries to get mu's for subsets of data
                                mu = self.mu # How things originally were when I was converting curves from only aves from B10K
                                # Convert scaled time to real time
                                df[self.time_field] = df[self.time_field] / mu  # Convert to generations
                                for key in self.gen_time_dict.keys():  # Step can be improved if keys list is sorted
                                    if key in filename:
                                        generation_time = self.gen_time_dict[key]
                                        df[self.time_field] = df[self.time_field] * generation_time
                                # print(df)
                                # # [OLD LOCATION FOR MINMAX NORM] Convert Coalescence Rate to Ne
                                df[self.value_field] = 1 / df[self.value_field]  # Take inverse of coalescence rate
                                df[self.value_field] = df[self.value_field] / (2 * mu)
                            # Drop ignored fields
                            if self.ignore_fields and len(self.ignore_fields) > 0:
                                assert all(field in df.columns for field in self.ignore_fields), "Not all fields are in df.columns. Check ignore_fields"
                                df = df.drop(self.ignore_fields, axis=1)
                            mySeries.append(df)
                            namesofMySeries.append(filename[:-len(suff)])
                    else:
                        print(f"self.suffix: {self.suffix} does not match suffix of {filename} {filename[-len(suff):]}")
                else:
                    # print("We got a directory")
                    for filename in os.listdir(directory + subdir + "/"):  # depending on filename's taxanomical class, mu may vary
                        if suff == filename[-len(suff):]:
                        # if filename.endswith(suff):  # Check for correct file suffix
                            if filename[:-len(suff)] not in self.to_omit:
                                df = pd.read_csv(directory + subdir + "/" + filename, **read_csv_kwargs)
                                # Create index of df
                                if not self.time_field:  # We need to impute index and a fake right_time_boundary col
                                    impromptu_index = list(range(df.shape[0]))
                                    df.index = impromptu_index
                                else:
                                    df.set_index(keys=[self.index_field], inplace=True)
                                df = df.iloc[self.omit_front_prior:len(df)-self.omit_back_prior]  # Perform omission of points prior to saving in self.mySeries
                                df.sort_index(inplace=True)
                                self.subdir2file_dict[subdir].append(filename) # GOOD FOR CLUSTERING WITH BIRDS AND MAMMALS WHICH HAVE DIFF MU's.
                                if real_time:  # If real time curves are desired, transform current df
                                    mu = self.class_mu_dict[self.subdir_class_dict[subdir]]  # Index into my convoluted ass dictionaries to get mu's for subsets of data
                                    # Convert scaled time to real time
                                    df[self.time_field] = df[self.time_field] / mu  # Convert to generations
                                    for key in self.gen_time_dict.keys():  # Step can be improved if keys list is sorted
                                        if key in filename:
                                            generation_time = self.gen_time_dict[key]
                                            df[self.time_field] = df[self.time_field] * generation_time
                                    # Convert Coalescence Rate to Ne
                                    df[self.value_field] = 1 / df[self.value_field]  # Take inverse of coalescence rate
                                    df[self.value_field] = df[self.value_field] / (2 * mu)
                                # Drop ignored fields
                                if self.ignore_fields and len(self.ignore_fields) > 0:
                                    assert all(field in df.columns for field in self.ignore_fields), "Not all fields are in df.columns. Check ignore_fields"
                                    df = df.drop(self.ignore_fields, axis=1)
                                mySeries.append(df)
                                namesofMySeries.append(filename[:-len(suff)])
                        else:
                            print(f"self.suffix: {self.suffix} does not match suffix of {filename} {filename[-len(suff):]}")
        # HERE MIGHT BE GOOD TO WINDOW OFF DATA
        if self.time_window:
            # print('len of my series before windowing:', len(mySeries))
            mySeries = windowMySeries(mySeries=mySeries,
                                      by=self.time_field,
                                      upperbound=self.upperbound,
                                      lowerbound=self.lowerbound)
            # print('len of my series after windowing:', len(mySeries))
        series_lengths = {len(series) for series in mySeries}  # Compile unique Series lengths
        series_lengths_list = [len(series) for series in mySeries]  # Compile unique Series length
        # print(series_lengths)
        # print(series_lengths_list)
        return mySeries, namesofMySeries, series_lengths, series_lengths_list

    def read_gen_times(self, directory: "str") -> "dict":
        '''
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

    def filter_data(self, uniform_ts_curve_domains=False):
        '''
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
        max_series_len = max(self.series_lengths)
        newMySeries = []
        newNamesOfMySeries = []
        for idx, series in enumerate(self.mySeries):
            if len(series) == max_series_len or self.time_window: # "or self.time_window" is included in condition for enabling the reading of time window'd data
                # print(self.time_field)
                # print(series)
                if series[self.time_field].iloc[0] == 0:  # If 1st time entry is 0
                    # print(f"len of trimmed df: {len(series.iloc[1:])}")
                    newMySeries.append(series.iloc[1:])  # Clip off 1st entry to avoid -inf err when scaling to log10 scale in normalize()
                    # Entry at 0th and 1st idx are identical for lambda so no meaningful info should be lost
                else:
                    newMySeries.append(series)
                newNamesOfMySeries.append(self.namesofMySeries[idx])
        self.mySeries = newMySeries
        self.namesofMySeries = newNamesOfMySeries

        # Find largest final time plotted among series
        # Make sure all series' are on the range of the series of the largest size (biggest final time recorded on X-axis)
        # This should be fine since all times boundaries are implied to end on inf anyways (found in original data)

        if uniform_ts_curve_domains:
            ts_beginning_time = {series[self.time_field].max() for series in self.mySeries} # units in terms of max(series) (Time)
            ts_beginning_time = max(ts_beginning_time)
            to_extend_idxs = [] # Record series' which had final recorded times less that biggest final time
            for i in range(len(self.mySeries)):
                if max(self.mySeries[i]) != ts_beginning_time: # If series doesn't extend to oldest time series beginning time
                    to_extend_idxs.append(i)
            for idx in to_extend_idxs: # Scale time boundaries of each series to the oldest known one in all data
                self.mySeries[idx][self.time_field].iloc[-1] = ts_beginning_time
                
        for idx, name in enumerate(self.namesofMySeries):
            self.name2series[name] = self.mySeries[idx]

    def normalize_series(self, series: "pd.series") -> "pd.series":
        '''
        Normalize a pandas series (df or col) to [0, 1].
        Handle case where all data may be 0
        '''
        # print("Normalizing")
        # print(series.unique())
        # print(series.unique()[0])
        if len(series.unique()) == 1:
            self.flat_curves += 1
            return np.zeros(len(series))
        else:
            return (series - series.min()) / (series.max() - series.min())

    def normalize(self):
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
        for idx in range(len(self.mySeries)):
            if self.log_scale_time:
                self.plot_on_log_scale = True
                self.mySeries[idx][self.time_field] = np.log10(self.mySeries[idx][self.time_field])
            if self.normalize_lambda:
                self.mySeries[idx][self.value_field] = self.normalize_series(self.mySeries[idx][self.value_field])
            else:
                pass

    def plot_series(self, num_to_plot=None, cols=5, fs_x=50, fs_y=25):
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
                axs[i, j].step(x_list, y_list, 'g-', where="pre")
                axs[i, j].set_title(self.namesofMySeries[i*cols+j])
                axs[i, j].set_xlabel(self.xlabel)  # time
                axs[i, j].set_ylabel(self.ylabel)  # size
        fig.patch.set_facecolor('white')  # Changes background to white
        plt.show()

    def elbow_method(self, random_state=205, gamma=None, save_name="cluster-related-figures/elbow-method-plot.png", plot=False, low=2, high=20, save_to=None):
        '''

        Usage ex: 
        save_to = "MSMC-Exploratory-Analysis/results/figures/"
        instance.elbow_method(save_to=save_to)
        '''
        fig = plt.figure()
        plt.suptitle(f"gamma = {gamma}")
        elbow_data = []
        cleanSeries = [series for series in self.mySeries]
        for n_clusters in range (low, high+1):
            if not gamma is None:
                print(f"soft-DTW gamma = {gamma}")
                if self.algo == "kmeans":
                    print(f"gamma = {gamma}")
                    km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=False, 
                                          metric="softdtw", 
                                          metric_params={"gamma": gamma}, 
                                          dtw_inertia=True, 
                                          random_state=random_state)
                elif self.algo == "kshapes":
                    km = KShape(n_clusters=n_clusters, verbose=False, 
                                random_state=random_state) 
                else:
                    km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=False, 
                                          metric="softdtw", 
                                          metric_params={"gamma": gamma}, 
                                          dtw_inertia=True, 
                                          random_state=random_state)
            else:
                print("DTW")
                if self.algo == "kmeans":
                    km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=False, 
                                          metric="dtw", dtw_inertia=True, 
                                          random_state=random_state)
                elif self.algo == "kshapes":
                    km = KShape(n_clusters=n_clusters, verbose=False, 
                                random_state=random_state) 
                else:
                    km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=False, 
                                          metric="dtw", dtw_inertia=True, 
                                          random_state=random_state)
            y_pred = km.fit_predict(cleanSeries)
            elbow_data.append((n_clusters, km.inertia_))
        ax = pd.DataFrame(elbow_data, columns=['clusters', 'distance']).plot(x='clusters', y='distance')
        self.elbow_data = elbow_data
        ax.set_ylabel("Distortion in Sum of squared distances (Inertia)")
        if save_to:
            plt.savefig(save_to + save_name, dpi=300)
        plt.show()
        return

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
        Func clusters using DBSCAN algorithm with DTW distance

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
        self.clustering_data = cleanSeries_with_df
        # for idx, name in enumerate(self.namesofMySeries):
        #     self.name2series[name] = self.mySeries[idx]
        # Compute some cluster barycenters
        gamma = metric_params["gamma"]
        self.label2barycenter = self.find_cluster_barycenters(iter, gamma)
        # Plots curves within their assigned clusters
        if plot_everything:
            self.cluster_count = len(set(self.dtw_labels))
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

    def plot_curve_clusters(self,
                            cleanSeries,
                            cols=3,
                            fs_x=25,
                            fs_y=50,
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
        # plot_count = math.ceil(math.sqrt(self.cluster_count))
        num_to_plot = self.cluster_count  # Set no. plots to num clusters
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
                             K = {self.cluster_count}')
            else:
                fig.suptitle(f'soft-DTW Clusters of {self.suptitle}\n{metric_params}, K = {self.cluster_count}')
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
                        if self.plot_on_log_scale and not self.log_scale_time:
                            x = np.log10(cleanSeries[i][self.time_field].to_numpy()) # Index mySeries for df
                        else:
                            x = cleanSeries[i][self.time_field].to_numpy()
                        y = cleanSeries[i][self.value_field].to_numpy()


                        # Curve color assignment
                        if self.namesofMySeries[i]+".txt" in self.subdir2file_dict["Archive"]: # Color Archive files differently (mammals)
                            reg_curve_color = "green" 
                        elif self.namesofMySeries[i]+".txt" in self.subdir2file_dict["birds_part_2"]:
                            reg_curve_color = "gray" 
                        elif self.namesofMySeries[i]+".txt" in self.subdir2file_dict["birds_part_1"]:
                            reg_curve_color = "gray" 
                        else:
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
                    if self.real_time and plot_iceages:  # Here is where to edit for differentiating curves by color
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
                    if self.real_time and plot_iceages:
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
        cluster_c = [len(self.dtw_labels[self.dtw_labels==i]) for i in range(self.cluster_count)]
        cluster_n = ["Cluster "+str(i) for i in range(self.cluster_count)]
        plt.figure(figsize=(15,5))
        plt.title(f"Cluster Distribution for {self.algo}")
        plt.bar(cluster_n,cluster_c)
        if save_to:
            plt.savefig(save_to + save_name, dpi = 100)
        plt.show()


    def plot_curve(self, name=None, df = None, dir = None, dups=0, stretch=0, err=0, thresh_min=None, thresh_max=None, winStart=None, winEnd=None, fs_x=10, fs_y=10, xlim_start=None, xlim_end=None, ylim_start=None, ylim_end=None,  save_to = None, additional_info=None, plot_on_log_scale=True):
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

        if plot_on_log_scale and not self.log_scale_time:
            x = np.log10(series[self.time_field].to_numpy()) # Index mySeries for df
            # if winStart and winEnd:
            #     winStart = np.log10(winStart)
            #     winEnd = np.log10(winEnd)
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
