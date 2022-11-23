"""
I am just using this as a test bed for now, please excuse the mess...
"""

# %%
from data_analysis.monte_carlo import ReviewProb
from data_analysis.rating_analysis import RatingAnalysisFriends, RatingAnalysisGeneral
from common.config_paths import (
            YELP_REVIEWS_PATH, YELP_USER_PATH, 
            MC_RESULTS_PATH, RESULTS, RATINGS_CORR_PATH)
from common.constants import restaurant_categories
from utils.query_raw_yelp import QueryYelp as qy
from data_analysis.correlation_analysis import get_pearson, get_linear_reg
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import math
from utils.plotting import (get_pkl_path, get_mc_pkl_path, 
                            bin_data, plot_bins, plot_mc_prob, 
                            plot_over_time,
                            plot_rating_dist,
                            plot_ratings,
                            get_time_periods)

time_periods = get_time_periods(num_periods=4)

# %%
PATH_PKL = "results/mc_final_redo/" # need Non-normalized versions
for t in time_periods:
    path, t_s = get_pkl_path(PATH_PKL, t)
    print(path("_prob_X"))

    rp = ReviewProb(save_path=PATH_PKL)
    rp.prep_data_range(date_range=(t[0], t[1]))
    p0, p1 = rp.get_probs(plot=True, save=True, normalize=False)
    
#%%
MC_PATH_PKL = "results/monte_carlo_prob0/"
MC_PATH_MEDIA = "media/monte_carlo_prob01_binned/"
bins = [x for x in range(0,51,5)]
ignore_exact = [0,1]
for t in time_periods:
    path, t_s = get_pkl_path(MC_PATH_PKL, t)
    print(path("_prob_X"))
    
    data_0 = pickle.load(open(path("_prob_0"), "rb"))
    data_1 = pickle.load(open(path("_prob_1"), "rb"))
    
    bd_0 = bin_data(data_0, bins, ignore_exact)
    bd_1 = bin_data(data_1, bins, ignore_exact)
    
    plot_bins(bd_1, bd_0)
    
    plt.xlabel("Number of i friends who reviewed same business")
    plt.ylabel("Monte Carlo probability")
    plt.ylim(0,1)
    plt.title(f"{t_s}: Ignoring i={ignore_exact}")
    
    i_str = "".join([str(x) for x in ignore_exact])
    # plt.savefig(f"{MC_PATH_MEDIA}{t_s}_prob_01_i{i_str}.png")
    plt.show()
    plt.clf()
# %%
