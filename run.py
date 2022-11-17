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


# IDEA: weight people with more friends more heavily -> they give us more information about probability of a review
# only focus on people with significant number of friends
# USERS['E6ATm0wReAmoFZjs_jHKcQ']


# %% Loading the ReviewProb object from pickle
pickle_path = "results/monte_carlo/ReviewProb_10000.pkl"
pickle_path = "results/monte_carlo/ReviewProb_ALL.pkl"


rb = pickle.load(open(pickle_path, "rb"))

plot
# %%
