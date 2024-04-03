"""This script contains supplementary functions for dFIC projest data anaysis, written by J.Stasinski unless stated otherwise"""
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import csv
from scipy.stats import ks_2samp, ks_1samp
from scipy.stats import skew, kurtosis
from scipy.stats import linregress


def abs_dist_by2(x, y):
    return np.abs(x-y)/2


def get_peak_freq(PSP, sampling_freq=1000, nperseg=2048):
    # Calculate peak frequency using Welch's method
    psp_f, psp_pxx = sig.welch(PSP, fs=sampling_freq, nperseg=nperseg, axis=0, scaling='density')
    psp_f = psp_f[:60]
    psp_pxx = psp_pxx[:60]
    psp_peak_freq = psp_f[np.argmax(psp_pxx, axis=0)]
    return psp_peak_freq, (psp_f, psp_pxx)


def calc_FC(time_series):
    FC = np.corrcoef(time_series.T)
    FC[np.diag_indices_from(FC)] = 0. # changing the diagonal (self_connections) to 0.
    #FC = np.triu(FC)
    return FC

def prep_FC(FC):
    FC[np.diag_indices_from(FC)] = 0. # changing the diagonal (self_connections) to 0.
    idxs = np.triu_indices_from(FC, k=1)
    ##choosing only upper traingle of FC matrix 
    triu_FC = np.asarray(FC[idxs])
    return triu_FC

def key_as_number(key):
    try:
        return int(key)
    except ValueError:
        return key



def BandPass(fs, lowcut=None, highcut=None, order=5, filt=sig.butter):
    '''written by H.Taher'''
    nyq = 0.5 * fs
    low = None
    high = None
    btype = None
    if (lowcut is not None):
        low = lowcut / nyq
    if (highcut is not None):
        high = highcut / nyq
    if (low and high):
        Wn = [low, high]
        btype = "bandpass"
    elif (low):
        Wn = low
        btype = "highpass"
    elif (high):
        Wn = high
        btype = "lowpass"
    b, a = filt(order, Wn, btype=btype)
    return b, a



def BandPassFilter(data, fs, lowcut=None, highcut=None, axis=0, filt=sig.butter, order=1, **kwargs):
    '''written by H.Taher'''
    b, a = BandPass(fs, lowcut, highcut, order=order, filt=filt)
    zi = sig.lfilter_zi(b, a)
    zi = zi[:, np.newaxis]
    y, z0 = sig.lfilter(b, a, data, axis=axis, zi=zi*data[0], **kwargs)
    return y

def FCFromTimeSeries(bv):
    '''written by H.Taher'''
    FC = np.corrcoef(bv, rowvar=False)
    return FC

def FCDFromTimeSeries(bv, window_size=30, sliding_increment=2):
    '''written by H.Taher'''
    u_idx = np.triu_indices(bv.shape[1], k=1)
    FCus = list()

    n_windows = np.ceil(bv.shape[0] / window_size).astype(int)
    bv = bv[:n_windows * window_size]
    # bv = bv.astype(np.float32)

    for i in range(0, bv.shape[0] - window_size, sliding_increment):
        sl = slice(i, i + window_size)
        FCD = np.corrcoef(bv[sl], rowvar=False, dtype=np.float32)
        np.fill_diagonal(FCD, 0)
        #FCD = np.log((1 + FCD) / (1 - FCD)) / 2
        FCus.append(FCD[u_idx])

    FCD = np.corrcoef(FCus, rowvar=True, dtype=np.float32)
    return FCD


def FCDDistribution(FCD, bins=np.linspace(-1, 1, 101), *args, **kwargs):
    """
    Compute the FCD distribution from a given FCD matrix.
    The function computes the upper triangular values of the FCD matrix and returns their histogram.
    Parameters:
        FCD (numpy.ndarray): The Functional Connectivity Density matrix.
        bins (numpy.ndarray or int, optional): Bins for histogram. Default is linspace from 0 to 1 with 101 bins.
    Returns:
        tuple: A tuple containing two elements:
            - numpy.ndarray: The histogram values.
            - numpy.ndarray: The bin edges.
    """
    u_idx = np.triu_indices_from(FCD, k=1)
    upper_vals = FCD[u_idx].flatten()
    return np.histogram(upper_vals, bins, density=True, *args, **kwargs)


def FCDStats(FCD, bins=np.linspace(0, 1, 101), if_mat=True, *args, **kwargs):
    """
    The function computes the upper triangular values of the FCD matrix and calculates statistics and KL divergence.
    Parameters:
        FCD (numpy.ndarray): The Functional Connectivity Density matrix.
        bins (numpy.ndarray or int, optional): Bins for histogram. Default is linspace from 0 to 1 with 101 bins.
    Returns:
        dict: A dictionary containing the following statistics and KL divergence:
            - 'mean': Mean of the upper values.
            - 'variance': Variance of the upper values.
            - 'skewness': Skewness of the upper values.
            - 'kurtosis': Kurtosis of the upper values.
    """
    if if_mat:
        u_idx = np.triu_indices_from(FCD, k=1)
        vals = FCD[u_idx].flatten()
    else:
        vals = FCD

    # Calculate statistics
    mean = np.mean(vals, *args, **kwargs)
    variance = np.var(vals, *args, **kwargs)
    skewness = skew(vals, *args, **kwargs)
    kurt = kurtosis(vals, *args, **kwargs)

    # Create and return a dictionary with computed values
    stats_dict = {
        'mean': mean,
        'var': variance,
        'skew': skewness,
        'kurt': kurt
    }
    return stats_dict

def compute_ks_distance(data1, data2, one_samp=False):
    """
    Compute the Kolmogorov-Smirnov (KS) distance between two data samples( Dynamical Functional Connectivity (dFC) matrices). 
    or one sample and a distribution
    Parameters:
    dfc_matrix1, dfc_matrix2: NumPy arrays representing dFC matrices for two datasets.
    Returns:
    1 - ks_distance: KS distance between the two dFC matrices.
    """
    if not one_samp:
    # Extract upper triangles
        data1 = prep_FC(data1)
        data2=  prep_FC(data1)
    # Compute the KS statistic and p-value
    ks_statistic, pval = ks_2samp(data1 , data2)

    return 1-ks_statistic, pval


def filter_Rmat(Rmat, MOUTmat, threshold=None):
    filt_Rmat = Rmat.copy()
    for r in range(Rmat.shape[0]):
        for c in range(Rmat.shape[1]):
            if MOUTmat[r,c] > threshold:
                filt_Rmat[r,c] = 0
    return filt_Rmat


def calc_windows(data, window_size, step_size):
    """
    Manually implement the sliding window logic.

    Parameters:
    - data: The input data array.
    - window_size: The size of the sliding window.
    - step_size: The step size for sliding the window.
    
    Returns:
    - windows: A list of windows extracted from the data.
    """
    windows = []
    for i in range(0, data.shape[0] - window_size + 1, step_size):
        window = data[i:i+window_size]
        windows.append(window)
    print(len(windows))
    return windows


from collections import Counter

def regimes_counter(pp, cut_off=6, perc_td=7.5):
    """
    Count and analyze regimes in a time series data.

    This function takes a list of values `pp`, representing a time series, and calculates the occurrences and percentages
    of different regimes in the time series. Regimes are categorized into three types: 'FP' (Fixed Point),
    'SLC' (Slow Limit Cycle), and 'FLC' (Fast Limit Cycle).

    Parameters:
    pp (list): A list of numeric values representing a time series.
    cut_off (int, optional): A threshold value to determine 'FP', 'SLC', and 'FLC'. Default is 6.
    perc_td (int, optional): The percentage threshold to identify different regimes in the time series. Default is 5.

    Returns:
    dict: A dictionary containing the counts and percentages of each regime in the time series.
    int: The number of different regimes that meet the percentage threshold.
    """

    # Initialize a dictionary to count different regimes
    p_dict = {reg: 0 for reg in ['FP', 'SLC', 'FLC']}

    # Loop through the time series to count regimes
    for index in range(len(pp) - 1):
        if pp[index] < cut_off and pp[index + 1] < cut_off:
            p_dict['FP'] += 1
        elif pp[index] > cut_off and pp[index + 1] > cut_off:
            p_dict['FLC'] += 1
        else:
            p_dict['SLC'] += 1

    # Calculate percentages for each regime
    for regime in p_dict:
        p_dict[regime] = (p_dict[regime] / (len(pp) - 1)) * 100

    # Count the number of different regimes that meet the percentage threshold
    c = 0
    for value in p_dict.values():
        if value >= perc_td:
            c += 1

    return p_dict, c


def poincare_analysis(psps, node_order, ax, o_par=50, node_dict=None):
# The poincare_analysis function performs a Poincaré analysis on a time-series data. 
# It calculates the intersection points of the time-series data and plots a Poincaré map.
# It also counts the number of regimes per node based on a specified cut-off line.
    if not node_dict: node_dict = {}
    reg_reg = {}
    N_nodes = psps.shape[1]
    for idx, n in enumerate(node_order[:N_nodes:1]):
        # Find local minima or maxima as potential intersection points
        local_extrema_indices = sig.argrelextrema(psps[:,n], np.greater, order=o_par)  # Change np.less to np.greater for maxima
        # Extract intersection points from the time-series data
        poincare_points = psps[:,n][local_extrema_indices]
        # count the regimes per node based on cut_off line:
        cut_off = 6.2
        p_dict, c = regimes_counter(poincare_points, cut_off)
        # store them in dict
        node_dict[n] = p_dict
        reg_reg[n] = c
        # Plot the Poincaré map
        ax.scatter(poincare_points[:-1], poincare_points[1:], s=40, marker ='.',color=cmap84(n), alpha=0.55)
        ax.set_xlabel('X at t')
        ax.set_ylabel('X at t+1')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        #np.save(DATA_LOC + f'regimes_node_dict_{node_dict["sim"]}_A.npy', node_dict, allow_pickle=True)
    
    reg_dict = Counter()

    for node in node_dict.keys():
        for reg in node_dict[node]:
            reg_dict[reg] += node_dict[node][reg] / N_nodes
        
    reg_dict = {reg:np.round(reg_dict[reg], 3) for reg in reg_dict.keys()}

    reg_count = Counter()
    for node in reg_reg.keys():
        if reg_reg[node] == 1: reg_count[1] += 1
        if reg_reg[node] == 2: reg_count[2] += 1
        if reg_reg[node] == 3: reg_count[3] += 1

    return reg_dict, reg_count, node_dict, ax
    
def FC_bootstrapping(r_dict, window_sizes=[1200, 1600, 2000], step_size=3, verbose=False):
    w_FCs = {}
    rejects = {}
    for ydx, y0t in enumerate(r_dict.keys()):
        w_FCs[y0t] = {}
        for gdx, gc in enumerate(r_dict[y0t].keys()):
            data = r_dict[y0t][gc][0]
            w_FCs[y0t][gc] = {}
            full_FC = r_dict[y0t][gc][2]
            avg_big_FC = np.mean(prep_FC(full_FC))
            if avg_big_FC >= 0.2: rejects[avg_big_FC] = (y0t,gc)
            full_R = r_dict[y0t][gc][3]
            for win_size in window_sizes:
                w_FCs[y0t][gc][win_size] = {}
                windows = calc_windows(data, win_size, step_size)
                
                for widx, window in enumerate(windows):
                    sFC = calc_FC(window)
                    avg_sFC = np.mean(prep_FC(sFC))
                    
                    # Assuming emp_FC is defined elsewhere or passed as an argument
                    simR = np.corrcoef(prep_FC(sFC), prep_FC(full_FC))[0,1]
                    empR = np.corrcoef(prep_FC(sFC), prep_FC(emp_FC))[0,1]
                    diff_R = empR - r_dict[y0t][gc][3]

                    if avg_big_FC >= 0.2:
                        empR_filt = 0
                        diff_R_filt = 0
                    else:
                        empR_filt = np.corrcoef(prep_FC(sFC), prep_FC(emp_FC))[0,1]
                        diff_R_filt = empR_filt - full_R
                        if np.abs(diff_R_filt) > 0.10: print('!!!', y0t, gc, win_size, diff_R_filt, empR_filt, r_dict[y0t][gc][3])

                    if verbose:
                        if (y0t == '0.004' or '0.1') and gc == '12' and widx in range(110,130):
                            print(y0t, gc, win_size, widx, ' ----> ', empR, diff_R, empR_filt, diff_R_filt)
                    w_FCs[y0t][gc][win_size][widx] = [simR, empR, diff_R, empR_filt, diff_R_filt, full_R ]
                    w_FCs[y0t] = dict(sorted(w_FCs[y0t].items(), key=lambda x: key_as_number(x[0])))
    w_FCs = dict(sorted(w_FCs.items(), key=lambda x: key_as_number(x[0])))
    return w_FCs, rejects

def FCD_bootstrapping(r_dict,  window_sizes = [1200, 1600, 2000], step_size=3, verbose=False):
    # Define the window sizes to iterate over
    fcd_dict = {}
    # Iterate over the keys in the r_dict dictionary
    for ydx, y0t in enumerate(r_dict.keys()):
        fcd_dict[y0t] = {}
        for gdx, gc in enumerate(r_dict[y0t].keys()):
            data = r_dict[y0t][gc][0]
            fcd_dict[y0t][gc] = {}
            # For each window size, calculate the windows and FCs
            print(y0t, gc)
            for win_size in window_sizes:
                fcd_dict[y0t][gc][win_size] = {}
                windows = calc_windows(data, win_size, step_size)
                for widx, window in enumerate(windows):
                    fcd = FCDFromTimeSeries(window, FCDwin, FCDinc)
                    fcd_dist, fcd_dist_bins = FCDDistribution(FCD=fcd, bins=np.linspace(-1, 1, 101))
                    fcd_dict[y0t][gc][win_size][widx] = {}
                    #fcd_dict[y0t][gc][win_size][widx]['fcd'] = fcd
                    fcd_dict[y0t][gc][win_size][widx]['fcd_dist'] = fcd_dist

                    sd_norm = np.sort(fcd_dist[50:] / np.max(fcd_dist[50:]))

                    ks = compute_ks_distance(sd_norm,  ed_norm,  one_samp=True)[0]
                    fcd_dict[y0t][gc][win_size][widx]['ks'] = ks
        
        np.save(DATA_LOC + f'FCD_bootstrapping_postFIC_{y0t}data.npy', fcd_dict[y0t], allow_pickle=True)
        
    return fcd_dict