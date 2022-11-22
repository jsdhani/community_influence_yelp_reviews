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


# %%
covid_range = (pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01'))
len_covid = covid_range[1] - covid_range[0] # 609 days
time_periods = []
for x in range(10):
    start = covid_range[0] - (x * len_covid)
    end = covid_range[1] - (x * len_covid)
    time_periods.append((start, end))
    
PKL_FOLDER_PATH = f"{RESULTS}/monte_carlo_prob0/"


# %% monte carlo
for t in time_periods:
    rp = ReviewProb(save_path=PKL_FOLDER_PATH)
    rp.prep_data_range(date_range=(t[0], t[1]))
    rp.get_probs(plot=False, save=True)

# # %% getting data
# for t in time_periods:
#     ra = RatingAnalysisGeneral(save_path=PKL_FOLDER_PATH)
#     ratings = ra.get_business_rating(date_range=(t[0], t[1]), 
#                  filter=None, 
#                  exclude_filter=False)
    
#     ra.save_ratings()
#     # ra.plot_ratings(title=
#     #                 "{} - {}".format(t[0].strftime('%Y-%m-%d'), 
#     #                       t[1].strftime('%Y-%m-%d')))
    

# %% correlation analysis
pears = []
lines = []
PATH = lambda x: f"{PKL_FOLDER_PATH}/{x}.pkl"

fig_save_path = 'media/'+ PKL_FOLDER_PATH.split('/')[-2]+ '/'

print("{:25}|{:^10}|{:^10}|{:^10}|{:^10}".format("Period", "coeff",
                                                 "p-value", "slope",
                                                 "intercept"))
print("-"*55)
for t in time_periods:
    t_s = f"{t[0].strftime('%Y-%m-%d')}_{t[1].strftime('%Y-%m-%d')}"
    path_prob = lambda x: PATH(f"{t_s}_prob_{x}")
    data_0 = pickle.load(open(path_prob(0), 'rb'))
    data_1 = pickle.load(open(path_prob(1), 'rb'))
    
    data = data_0
    
    pear_0 = get_pearson(data_0, alt="two-sided")
    line_0 = get_linear_reg(data_0)
    
    pear_1 = get_pearson(data_1, alt="two-sided")
    line_1 = get_linear_reg(data_1)
    
    pear = (pear_1[0],#-pear_1[0], 
            pear_1[1])#-pear_1[1])
    line = (line_1[0],#-line_1[0], 
            line_1[1])#-line_1[1])
    
    pears.append(pear)
    lines.append(line)
    
    print("{:25}|{:^10.3}|{:^10.3}|{:^10.3}|{:^10.3}".format(t_s, pear[0], pear[1], line[0], line[1]))

#%% plotting corr coeff over time
pears = np.array(pears)
y_pos = list(range(len(pears)))
fig, ax = plt.subplots()
ax.bar(y_pos, pears[:,0][::-1])
_=ax.set_xticks(y_pos, labels=[f"{p[1].strftime('%Y')}" for p in time_periods][::-1])
if fig_save_path: plt.savefig(fig_save_path+'all_corr.png')

#%% ploting linear regression slopes over time
plt.clf()
lines = np.array(lines)
y_pos = list(range(len(lines)))
fig, ax = plt.subplots()
ax.bar(y_pos, lines[:,0][::-1])
_=ax.set_xticks(y_pos, labels=[f"{p[1].strftime('%Y')}" for p in time_periods][::-1])
if fig_save_path: plt.savefig(fig_save_path+'all_lin.png')


# %% plotting with linear regression
for i,t in enumerate(time_periods):
    t_s = f"{t[0].strftime('%Y-%m-%d')}_{t[1].strftime('%Y-%m-%d')}"
    path_prob = lambda x: PATH(f"{t_s}_prob_{x}")
    data = pickle.load(open(path_prob(0), 'rb'))
    data = np.array([[x,y] for x,y in data.items()])
    
    plt.clf()
    plt.xlabel("User Rating")
    plt.ylabel("Avg. Friend Rating")
    line = lines[i]
    plt.plot(data[:,0], line[0]*data[:,0] + line[1], label=t_s)

    plt.scatter(data[:,0], data[:,1])
    plt.xlabel("User Rating")
    plt.ylabel("Average Friend Rating")
    plt.title(t_s)
    plt.legend()
    
    
    if fig_save_path: plt.savefig(fig_save_path+t_s+'.png')

# %%
