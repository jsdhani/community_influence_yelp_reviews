"""
I am just using this as a test bed for now, please excuse the mess...
"""

# %%
from data_analysis.monte_carlo import ReviewProb
from data_analysis.rating_analysis import RatingAnalysisFriends, RatingAnalysisGeneral
from common.config_paths import (YELP_BUSINESS_PATH,
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

time_periods = get_time_periods(num_periods=2)
PATH_PKL = "results/mc_final_redo/" 
# PATH_MEDIA = "media/monte_carlo_hetrogenityQ/CA/"
SAVE_FIG_PATH = "media/final_paper_figures/"
#%%
# for t in time_periods:
#     path, t_s = get_pkl_path(PATH_PKL, t) # gets the correctly formatted path
#     print(path("_prob_X"))

#     rp = ReviewProb(save_path=PATH_PKL)
#     rp.prep_data_range_region(date_range=(t[0], t[1]))
#     p0, p1 = rp.get_probs(plot=False, save=True, normalize=False)


#%%
bins = [x for x in range(0,51,5)]
# bins = [0,0,1]
ignore_exact = [0,1,2,3]
i_str = "".join([str(x) for x in ignore_exact])
print(i_str)
p = [] # [[P(0|i=0), P(0|i>0)], [P(1|i=0), P(1|i>0)]]
h = []
for t in time_periods:
    path, t_s = get_pkl_path(PATH_PKL, t)
    print(path("_prob_X"))
    
    data_0 = pickle.load(open(path("_prob_0"), "rb"))
    data_1 = pickle.load(open(path("_prob_1"), "rb"))
    
    # normalize before binning
    # data_0 = {k:v/sum(data_0.values()) for k,v in data_0.items()}
    # data_1 = {k:v/sum(data_1.values()) for k,v in data_1.items()}
    
    bd_0 = bin_data(data_0, bins, ignore_exact, normalize=True)
    bd_1 = bin_data(data_1, bins, ignore_exact, normalize=True)
    
    h0, h1 = plot_bins(bd_0, bd_1)
    h.append([h0, h1])
        
    plt.xlabel("Number of i friends who reviewed same business")
    plt.ylabel("Monte Carlo probability")
    plt.title(f"{t_s}: Ignoring i={ignore_exact}")
    plt.ylim(0,1)
    
    # plt.savefig(f"{MC_PATH_MEDIA}{t_s}_prob_01_i{i_str}.png")
    # plt.show()
    # plt.clf()
    
    # p.append([[bd_0[0], bd_0[1]],
    #           [bd_1[0], bd_1[1]]])

extra = f"\n(ignoring i={ignore_exact})" if ignore_exact else ""
plt.title(f"MC probability distribution for user reviews {extra}")
plt.legend([h[1][0], h[1][1], h[0][0],h[0][1]],
    ['PRE-COV: P(0,i)', 'PRE-COV: P(1,i)', 'COVID: P(0,i)', 'COVID: P(1,i)', ])
i_str = "".join([str(x) for x in ignore_exact])
plt.savefig(f"{SAVE_FIG_PATH}/MC/MC_2018_2021_i{i_str}.png")
# %%
#p[:, 1, 0] # P(1|i=0)
#p[:, 1, 1] # P(1|i>0)
#p[:, 0, 0] # P(0|i=0)
# fig, ax = plt.subplots()
plot_over_time(p, time_periods, idx=0, 
                labels=('P(0|i=0)', 'P(1|i=0)'))
plt.ylim(0, 1.05)
plt.title("Probability of that a user does/doesn't write a review \nwhen no friends have reviewed over time")
# plt.savefig("media/final_ptt_plots/mc_final_redo/compare/p_ie0.png")
plt.show()
plt.clf()

plot_over_time(p, time_periods, idx=1,
                labels=('P(0|i>0)', 'P(1|i>0)'))
plt.ylim(0, 1.05)
plt.title("Probability of that a user does/doesn't write a review \nwhen at least 1 friend has reviewed over time")
# plt.savefig("media/final_ptt_plots/mc_final_redo/compare/p_ig0.png")
plt.show()
plt.clf()
# %%
