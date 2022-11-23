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
#%%

PKL_FOLDER_PATH = f"{RESULTS}ratings-no-rest/ratings_"
SAVE_PATH = "media/final_ptt_plots/ratings/friend_corr/no-rest/"


# %% plotting with linear regression
# # FOR RATINGS
for i,t in enumerate(time_periods):
    path, t_s = get_pkl_path(PKL_FOLDER_PATH, t)
    data = pickle.load(open(path(''), 'rb'))
    line = get_linear_reg(data)
    plot_ratings(data, t_s, section="No Restaurants",
                 line=line, save_path=SAVE_PATH)
    
    
#     if fig_save_path: plt.savefig(fig_save_path+t_s+'.png')

# %%
