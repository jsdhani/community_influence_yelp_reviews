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


# %% Loading the ReviewProb object from pickle
pickle_path_1 = "results/monte_carlo/ReviewProb_10000.pkl"
pickle_path_a = "results/monte_carlo/ReviewProb_ALL.pkl"


rb_1 = pickle.load(open(pickle_path_1, "rb"))
rb_a = pickle.load(open(pickle_path_a, "rb"))

# %% plotting the review probabilities

def plot_prob(prob, thresh=5, normalize=False):
    prob_sorted = sorted(prob.items())[thresh:]
    
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
plot_prob(rb_1)
# plt.show()
plt.title("All users")
plot_prob(rb_a, normalize=True)
plt.show()

# %%
