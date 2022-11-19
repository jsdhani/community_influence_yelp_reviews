# %%
from data_analysis.review_prob import ReviewProb
from common.config_paths import YELP_REVIEWS_PATH, YELP_USER_PATH, MT_RESULTS_PATH, RESULTS
from utils.query_raw_yelp import QueryYelp as qy
from data_analysis.correlation_analysis import get_pearson
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# collecting data for the monte carlo simulation starting from today, and going back every 609 days until we have 10 time periods
covid_range = (pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01'))
len_covid = covid_range[1] - covid_range[0] # 609 days
time_periods = []
for x in range(10):
    start = covid_range[0] - (x * len_covid)
    end = covid_range[1] - (x * len_covid)
    time_periods.append((start, end))
    
new_path = RESULTS + 'BIG_MONTE_CARLO/'

# display as a chart:
print("{:15}|{:^10}|{:^10}".format("Period", "coeff", "p-value"))
print("-"*35)
coeffs = []
PATH = lambda x: f"{new_path}ReviewProb_None-{x}.pkl"
for p in time_periods:
    c+=1
    # if c % 2 == 0: continue
    t_path = PATH(f"{p[0].strftime('%Y-%m-%d')}_{p[1].strftime('%Y-%m-%d')}")
    prob = pickle.load(open(t_path, 'rb'))
    pear = get_pearson(prob)
    period_str = f"{p[0].strftime('%Y-%m')}_{p[1].strftime('%Y-%m')}"
    coeffs.append(pear.statistic)
    print("{:15}|{:^10.3}|{:^10.3}".format(period_str, pear.statistic, pear.pvalue))
    
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