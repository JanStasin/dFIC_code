import random
import time
import numpy as np
import scipy.io as sio
import scipy.signal as sig
from tvb.simulator.lab import *
import matplotlib.pyplot as plt

import csv
import itertools
from tvb.basic.neotraits.api import NArray, List, Range, Final
from scipy.stats import zscore
from scipy import signal
import string
import pandas as pd
import matplotlib.patches as patches


DATA_LOC2 = '/Users/jansta/fic_jansen/stability_analysis/paper_sims/paper_hcp_sims/y0_072/'
PLOT_LOC = '/Users/jansta/fic_jansen/stability_analysis/paper_plots/II_plots/'

DATA_BS = '/Users/jansta/fic_jansen/stability_analysis/paper_sims/paper_hcp_sims/y0_072/bootstrap/'
#loading the full length mmf values:
full_mmf_mat = np.load(DATA_LOC2 + 'mmf_thresh_fit_map.npy', allow_pickle=True)
full_mmf_dif = full_mmf_mat[:12,:]- full_mmf_mat[12:,:]


gcs = [str(g) for g in range(30)]
y0s = [str(i) for i in [0.001, 0.004, 0.007, 0.01, 0.013, 0.016, 0.0189, 0.1, 0.11, 0.12, 0.13, 0.14]]

## loading mmf values per y0t, per gc, per window
mmf_comb_dict,  mmf_comb_dict_nf = np.load(DATA_BS + 'both_mmf_boots_dicts.npy', allow_pickle=True)
win_size=1600
load_data = False
if not load_data:
    perm_res  = {}

    for ydx, y0t in enumerate(y0s[4:]):
        
        perm_res[y0t] = {}
        
        for gdx, gc in enumerate(mmf_comb_dict[y0t].keys()):
            print(y0t,gc)
            perm_res[y0t][gc] = {}
            postFIC_bsMMFs = mmf_comb_dict[y0t][gc][win_size].values()
            noFIC_bsMMFs = mmf_comb_dict_nf[y0t][gc][win_size].values()


            full_mean_diff = np.mean(list(postFIC_bsMMFs) - np.mean(list(noFIC_bsMMFs)))
            full_median_diff = np.median(list(postFIC_bsMMFs) - np.median(list(noFIC_bsMMFs)))
            #print(full_mean_diff)

            MMF_pool = list(postFIC_bsMMFs) + list(noFIC_bsMMFs)
            if len(MMF_pool) == 0:
                print("No data available for", y0t, gc)
                continue
            print(len(MMF_pool))
            subset_size = int(len(MMF_pool) / 2)
            
            all_pairs = []
            all_med_pairs = []

            #extracting the difference in mmf value per param comb
            fmd = full_mmf_dif[list(mmf_comb_dict.keys()).index(y0t), list(mmf_comb_dict[y0t].keys()).index(gc) ]
            if fmd < 0 : print('!!!!!!!!!!!)')
        

            # Generate 100,000 pairs of subsets
            for p in range(1000000):

                # Shuffle the list to randomize the order of elements
                random.shuffle(MMF_pool)
                
                # Split the shuffled list into two subsets of 164 values each
                subset1 = MMF_pool[:subset_size]
                subset2 = MMF_pool[subset_size:]
                diff = np.mean(subset1) - np.mean(subset2)
                med_diff = np.median(subset1) - np.median(subset2)

                if p % 500000 == 0:
        
                    print(np.mean(subset1) , np.mean(subset2) )
                    #print(f'full R: {w_FCs[y0t][gc][2000][0][5]} - {w_FCs_nf[y0t][gc][2000][0][5]}')
                    print(f'{y0t , gc} full MMF diff, mean, median: {fmd , diff, med_diff}')
                    

                all_pairs.append(diff)
                all_med_pairs.append(med_diff)
            print(len(all_pairs))
            #perm_res[y0t][gc] = [all_pairs, w_FCs[y0t][gc][2000][0][5] - w_FCs_nf[y0t][gc][2000][0][5]]
            perm_res[y0t][gc] = [all_pairs, all_med_pairs, full_mean_diff, full_median_diff, fmd]
        np.save(DATA_BS + f'permutations_1m_v4_{y0t}.npy', perm_res[y0t], allow_pickle=True)
    #np.save(DATA_BS + f'permutations_100k_v4_03.npy', perm_res, allow_pickle=True)
##else:
    ##perm_res = np.load(DATA_BS + 'permutations_100k_v3.npy',  allow_pickle=True).item()