"""
I am just using this as a test bed for now, please excuse the mess...
"""

# %%
from data_analysis.monte_carlo import ReviewProb
from data_analysis.rating_analysis import RatingAnalysis
from common.config_paths import (
            YELP_REVIEWS_PATH, YELP_USER_PATH, 
            MC_RESULTS_PATH, RESULTS, RATINGS_CORR_PATH)
from utils.query_raw_yelp import QueryYelp as qy
from data_analysis.correlation_analysis import get_pearson, get_linear_reg
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


# %%
covid_range = (pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01'))
len_covid = covid_range[1] - covid_range[0] # 609 days
time_periods = []
for x in range(10):
    start = covid_range[0] - (x * len_covid)
    end = covid_range[1] - (x * len_covid)
    time_periods.append((start, end))

# %% getting data
# for t in time_periods:
#     ra = RatingAnalysis()
#     ra.prep_data(date_range=(t[0], t[1]))
    
#     ratings = ra.get_ratings()
#     ra.save_ratings()
#     ra.plot_ratings(title=
#                     "{} - {}".format(t[0].strftime('%Y-%m-%d'), 
#                                      t[1].strftime('%Y-%m-%d')))

    
# %% correlation analysis
print("{:25}|{:^10}|{:^10}|{:^10}".format("Period", "coeff", "p-value", "slope"))
print("-"*45)
pears = []
lines = []
PATH = lambda x: f"{RATINGS_CORR_PATH}ratings_{x}.pkl"
for t in time_periods:
    t_s = f"{t[0].strftime('%Y-%m-%d')}_{t[1].strftime('%Y-%m-%d')}"
    path = PATH(t_s)
    data = pickle.load(open(path, 'rb'))
    
    pear = get_pearson(data, alt="two-sided")
    line = get_linear_reg(data)
    
    pears.append((pear.statistic, pear.pvalue))
    lines.append(line)
    
    print("{:25}|{:^10.3}|{:^10.3}|{:^10.3}".format(t_s, pear[0], pear[1], line[0]))
    
#%%
pears = np.array(pears)
y_pos = list(range(len(pears)))
fig, ax = plt.subplots()
ax.bar(y_pos, pears[:,0][::-1])
_=ax.set_xticks(y_pos, labels=[f"{p[1].strftime('%Y')}" for p in time_periods][::-1])

#%% ploting linear regression
lines = np.array(lines)
y_pos = list(range(len(lines)))
fig, ax = plt.subplots()
ax.bar(y_pos, lines[:,0][::-1])
_=ax.set_xticks(y_pos, labels=[f"{p[1].strftime('%Y')}" for p in time_periods][::-1])


# %% plotting just covid data
for i,t in enumerate(time_periods[:3]):
    t_s = f"{t[0].strftime('%Y-%m-%d')}_{t[1].strftime('%Y-%m-%d')}"
    path = PATH(t_s)
    data = pickle.load(open(path, 'rb'))
    
    
    # plotting the data
    # plt.scatter(data[:,0], data[:,1])
    plt.title("Rating Influence")
    plt.xlabel("User Rating")
    plt.ylabel("Avg. Friend Rating")
    # plotting the lines from linear regression
    line = lines[i]
    plt.plot(data[:,0], line[0]*data[:,0] + line[1], label=t_s)

plt.legend()

# %%
