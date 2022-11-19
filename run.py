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
import pickle


# %%
covid_range = (pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01'))
len_covid = covid_range[1] - covid_range[0] # 609 days
time_periods = []
for x in range(10):
    start = covid_range[0] - (x * len_covid)
    end = covid_range[1] - (x * len_covid)
    time_periods.append((start, end))
    
for t in time_periods:
    ra = RatingAnalysis()
    ra.prep_data(date_range=(t[0], t[1]))
    
    ratings = ra.get_ratings()
    ra.save_ratings()
    ra.plot_ratings(title=
                    "{} - {}".format(t[0].strftime('%Y-%m-%d'), 
                                     t[1].strftime('%Y-%m-%d')))

    
# %%
