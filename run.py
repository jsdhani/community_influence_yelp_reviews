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
# Plotting rating distributions for businesses
path=f'{RESULTS}ratings-overall-no-rest/ratings_'
save_path='media/final_ptt_plots/ratings/distributions/no-rest/'

for i, t in enumerate(time_periods):
    curr_path, t_s = get_pkl_path(path, t)
    data = pickle.load(open(curr_path(""), 'rb'))

    plot_rating_dist(list(data.values()), t_s, 
                    section='Restaurant', alpha=.5)#, save_path=save_path)
    # plt.show()
    # plt.clf()
plt.title(f"Rating Distribution for Non-Restaurants")
plt.legend(loc='upper left')
plt.savefig(f'{save_path}overlay.png')
# %%
