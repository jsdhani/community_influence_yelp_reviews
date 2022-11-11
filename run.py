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
    
# %% Iterate through the reviews and create dictionary of user ids and their network

####### METHOD 1: iterating through businesses first #######
# USERS:      {ID: (friends_ID)} # friends_ID is a set of user ids for constant time lookup
# BUSINESSES: {ID: (Usr_ID)}  # Usr_ID is also a set

# Assuming we created the data structure above we now preform the following:
#    Note that this will miss P(User writes a review | 0 friends wrote a review)?
#        ^ This isn't actually true but I feel like Im still missing something related to this
prob_counts = {} # keeps track of instances of (User writes a review | i friend(s) wrote a review) where i is the key
for usr_ids in BUSINESSES:
    for id in usr_ids:
        num_rev = 0
        for friend_id in USERS[id]:
            if friend_id in usr_ids:
                num_rev += 1
        
        # add to the probabilities dictionary or create it
        if num_rev in prob_counts:
            prob_counts[num_rev] += 1
        else:
            prob_counts[num_rev] = 1

# in the worst case everyone is friends with everyone and so this would run in O(b*n^2) 
# where n is the number of users and b is the number of businesses
    
####### METHOD 2: iterating through users first #######
# USERS:    {ID: {"Network": (friends_ID), 
#               "Reviews": (business_ID)}
#           }

for usr in USERS:
    for b in usr["Reviews"]:
        num_rev = 0
        for friend_id in usr["Network"]:
            if b in USERS[friend_id]["Reviews"]: # constant time lookup
                num_rev += 1
        
        # add to the probabilities dictionary or create it
        if num_rev in prob_counts:
            prob_counts[num_rev] += 1
        else:
            prob_counts[num_rev] = 1

# this should also run in O(b*n) 
# will use this to validate that i havent missed anything in the first method

# iterate through users and put them into a dictionary with the key being the user id
    # and the value being another dict of business ids that they have reviewed
    # and the value of that dict is the review text
# this allows for constant time look up of the reviews of a user and business
