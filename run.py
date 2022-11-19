"""
I am just using this as a test bed for now, please excuse the mess...
"""

# %%
from data_analysis.review_prob import ReviewProb
from common.config_paths import YELP_REVIEWS_PATH, YELP_USER_PATH, MT_RESULTS_PATH, RESULTS
from utils.query_raw_yelp import QueryYelp as qy
from data_analysis.correlation_analysis import get_pearson, get_linear_reg
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# %% MEGA MONTE CARLO DATA COLLECTION TIME
# collecting data for the monte carlo simulation starting from today, and going back every 609 days until we have 10 time periods
covid_range = (pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01'))
len_covid = covid_range[1] - covid_range[0] # 609 days
time_periods = []
for x in range(8):
    start = covid_range[0] - (x * len_covid)
    end = covid_range[1] - (x * len_covid)
    time_periods.append((start, end))
    
new_path = RESULTS + 'BIG_MONTE_CARLO/'

# %% display the monte carlo data
# display as a chart:
print("{:15}|{:^10}|{:^10}".format("Period", "coeff", "p-value"))
print("-"*35)
coeffs = []
PATH = lambda x: f"{new_path}ReviewProb_None-{x}.pkl"
for p in time_periods:
    # if c % 2 == 0: continue
    t_path = PATH(f"{p[0].strftime('%Y-%m-%d')}_{p[1].strftime('%Y-%m-%d')}")
    prob = pickle.load(open(t_path, 'rb'))
    pear = get_linear_reg(prob)[0]
    period_str = f"{p[0].strftime('%Y-%m')}_{p[1].strftime('%Y-%m')}"
    coeffs.append(pear)
    print("{:15}|{:^10.3}|{:^10.3}".format(period_str, pear, ""))
    
    # also adding to plot:
    prob_sorted = sorted(prob.items())
    plt.plot([x[0] for x in prob_sorted], [y[1] for y in prob_sorted])
    
plt.legend([f"{p[0].strftime('%Y-%m')}_{p[1].strftime('%Y-%m')}" for p in time_periods])
plt.xlim(0,5)# limiting the x axis to 0 to 100
plt.show()

y_pos = list(range(len(coeffs)))
fig, ax = plt.subplots()
ax.bar(y_pos, coeffs[::-1])
ax.set_xticks(y_pos, labels=[f"{p[1].strftime('%Y')}" for p in time_periods][::-1])
#%% IDEA: weight people with more friends more heavily 
#   -> they give us more information about probability of a review
# only focus on people with significant number of friends
# USERS['E6ATm0wReAmoFZjs_jHKcQ']
# rp = ReviewProb()
# rp.prep_data_range(date_range=(pd.Timestamp.min, pd.Timestamp('2019-12-01')))
# rp.get_prob(plot=False, save=True, weighted=True)

# # %% Loading the ReviewProb object from pickle
# # pickle_path_1 = "results/monte_carlo/ReviewProb_10000.pkl"
# # pickle_path_a = "results/monte_carlo/ReviewProb_ALL.pkl"

# pkl_path_0 = 'results/monte_carlo/ReviewProb_None-1677-09-21_2019-12-01.pkl' # pre-covid entire time period
# pkl_path_1 = 'results/monte_carlo/ReviewProb_None-2018-04-01_2019-12-01.pkl' # pre-covid19 short time period (609 days)
# pkl_path_2 = 'results/monte_carlo/ReviewProb_None-2019-12-01_2021-08-01.pkl' # during covid19 (609 days)

# rb_0 = pickle.load(open(pkl_path_0, 'rb'))
# rb_1 = pickle.load(open(pkl_path_1, "rb"))
# rb_2 = pickle.load(open(pkl_path_2, "rb"))


# # %% Calc correlation
# pear_0 = get_pearson(rb_0)
# pear_1 = get_pearson(rb_1)
# pear_2 = get_pearson(rb_2)

# # display as a chart:
# print("{:12}|{:^10}|{:^10}".format("Period", "coeff", "p-value"))
# print("-"*32)
# print("{:12}|{:^10.3}|{:^10.3}".format("Pre-COVID-L", pear_0.statistic, pear_0.pvalue))
# print("{:12}|{:^10.3}|{:^10.3}".format("Pre-COVID-S", pear_1.statistic, pear_1.pvalue))
# print("{:12}|{:^10.3}|{:^10.3}".format("COVID", pear_2.statistic, pear_2.pvalue))

# # %% plotting the review probabilities

# def plot_prob(prob, thresh_low=1, thresh_high=10, normalize=False):
#     prob_sorted = sorted(prob.items())[thresh_low:-thresh_high]
    
#     # normalizing the values
#     if normalize:
#         total = sum(prob.values())
#         prob_sorted = [(k, v/total) for k, v in prob_sorted]
    
#     plt.scatter([x[0] for x in prob_sorted],
#                 [y[1] for y in prob_sorted])
#     plt.xlabel("friend counts")
#     plt.ylabel("Frequency")


# # %%
# plt.title("Prob vs friend count")
# plot_prob(rb_1, thresh_low=1)
# # plt.show()

# plt.title("Prob vs friend count")
# plot_prob(rb_2, thresh_low=1)
# plt.show()

# %%
