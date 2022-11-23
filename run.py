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
                            plot_rating_dist)


# %%
covid_range = (pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01'))
len_covid = covid_range[1] - covid_range[0] # 609 days
time_periods = []
for x in range(4):
    start = covid_range[0] - (x * len_covid)
    end = covid_range[1] - (x * len_covid)
    time_periods.append((start, end))
    
#%%
PKL_FOLDER_PATH = f"{RESULTS}/monte_carlo_prob0/"
PATH = lambda x: f"{PKL_FOLDER_PATH}/{x}.pkl"
fig_save_path = 'media/'+ PKL_FOLDER_PATH.split('/')[-2]+ '/'


# %% plotting rating distributions for businesses
path=f'{RESULTS}ratings-overall/ratings_'
save_path='media/final_ptt_plots/ratings/distributions/all/'

for i, t in enumerate(time_periods):
    curr_path, t_s = get_pkl_path(path, t)
    data = pickle.load(open(curr_path(""), 'rb'))
    
    plot_rating_dist(list(data.values()), t_s, 
                     section='Business', save_path=save_path)
    plt.show()
    plt.clf()

# %% plotting with linear regression
# # FOR RATINGS
# for i,t in enumerate(time_periods):
#     t_s = f"{t[0].strftime('%Y-%m-%d')}_{t[1].strftime('%Y-%m-%d')}"
#     path_prob = lambda x: PATH(f"{t_s}_prob_{x}")
#     data = pickle.load(open(path_prob(0), 'rb'))
#     data = np.array([[x,y] for x,y in data.items()])
    
#     plt.clf()
#     plt.xlabel("User Rating")
#     plt.ylabel("Avg. Friend Rating")
#     line = lines[i]
#     plt.plot(data[:,0], line[0]*data[:,0] + line[1], label=t_s)

#     plt.scatter(data[:,0], data[:,1])
#     plt.xlabel("User Rating")
#     plt.ylabel("Average Friend Rating")
#     plt.title(t_s)
#     plt.legend()
    
    
#     if fig_save_path: plt.savefig(fig_save_path+t_s+'.png')

# %%
