"""
I am just using this as a test bed for now, please excuse the mess...
"""

# %%
from data_analysis.review_prob import ReviewProb
from common.config_paths import YELP_REVIEWS_PATH, YELP_USER_PATH, MT_RESULTS_PATH
from utils.query_raw_yelp import QueryYelp as qy
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# IDEA: weight people with more friends more heavily 
#   -> they give us more information about probability of a review
# only focus on people with significant number of friends
# USERS['E6ATm0wReAmoFZjs_jHKcQ']

# prob before covid
rp = ReviewProb()
rp.prep_data_range(date_range=(pd.Timestamp('2018-04-01'), pd.Timestamp('2019-12-01')))
rp.get_prob(plot=False, save=True, weighted=True)

# prob after covid
rp = ReviewProb()
rp.prep_data_range(date_range=(pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01')))
rp.get_prob(plot=False, save=True, weighted=True)

# %% Loading the ReviewProb object from pickle
# pickle_path_1 = "results/monte_carlo/ReviewProb_10000.pkl"
# pickle_path_a = "results/monte_carlo/ReviewProb_ALL.pkl"

# pp_pre_cov19 = 'results\monte_carlo\ReviewProb_None-1677-09-21_2019-12-01.pkl'
# pp_preshort_cov19 = 'results\monte_carlo\ReviewProb_None-2018-04-01_2019-12-01.pkl'
# pp_cov19 = 'results\monte_carlo\ReviewProb_None-2019-12-01_2021-08-01.pkl'

pp_preshort_cov19 = 'results\monte_carlo\W-ReviewProb_None-2018-04-01_2019-12-01.pkl'
pp_cov19 = 'results\monte_carlo\W-ReviewProb_None-2019-12-01_2021-08-01.pkl'

rb_1 = pickle.load(open(pp_preshort_cov19, "rb"))
rb_2 = pickle.load(open(pp_cov19, "rb"))

# %% plotting the review probabilities

def plot_prob(prob, thresh_low=1, thresh_high=10, normalize=False):
    prob_sorted = sorted(prob.items())[thresh_low:-thresh_high]
    
    # normalizing the values
    if normalize:
        total = sum(prob.values())
        prob_sorted = [(k, v/total) for k, v in prob_sorted]
    
    plt.scatter([x[0] for x in prob_sorted],
                [y[1] for y in prob_sorted])
    plt.xlabel("friend counts")
    plt.ylabel("Frequency")
    
# %%
plt.title("1K users")
plot_prob(rb_1, thresh_low=1)
# plt.show()
plt.title("All users")
plot_prob(rb_2, thresh_low=1)
plt.show()

# %%
