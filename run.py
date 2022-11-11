"""
I am just using this as a test bed for now, please excuse the mess...
"""

# %%
from utils.query_raw_yelp import QueryYelp as qy
from data_analysis.review_prob import ReviewProb
from common.config_paths import YELP_REVIEWS_PATH, YELP_BUSINESS_PATH, YELP_USER_PATH
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# %% Get the data
b_reader = qy.get_json_reader(YELP_BUSINESS_PATH, chunksize=1000)

# Only getting 100 businesses for now:
CUTTOFF = 100

# %%
# All we need from the business is their ID so everything else can be ignored
b_ids = np.array([])
for chunk in b_reader:    
    b_df = chunk[["business_id","review_count"]].dropna()
    # Businesses with > 50 reviews captures 28K/150K businesses
    b_df = b_df[(b_df["review_count"] > 500)] # high review count businesses for now
    
    # sampling  businesses
    sample_size = min(CUTTOFF-len(b_ids), len(b_df))
    b_ids = np.append(b_ids, b_df["business_id"].sample(sample_size).values)
    
    if len(b_ids) >= CUTTOFF: # cut off
        break
    
# %% Iterate through the reviews and create dictionary of user ids and their reviews

# iterate through users and put them into a dictionary with the key being the user id
    # and the value being another dict of business ids that they have reviewed
    # and the value of that dict is the review text
# this allows for constant time look up of the reviews of a user and business