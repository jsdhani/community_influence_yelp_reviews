"""
I am just using this as a test bed for now, please excuse the mess...
"""

# %%
from utils.query_raw_yelp import QueryYelp as qy
from data_analysis.review_prob import ReviewProb
from common.config_paths import YELP_REVIEWS_PATH, YELP_BUSINESS_PATH, YELP_USER_PATH
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle


# # %% Get the data
# b_reader = qy.get_json_reader(YELP_BUSINESS_PATH, chunksize=1000)

# # Only getting 100 businesses for now:
# CUTTOFF = 100

# # %%
# # All we need from the business is their ID so everything else can be ignored
# b_ids = np.array([])
# for chunk in b_reader:    
#     b_df = chunk[["business_id","review_count"]].dropna()
#     # Businesses with > 50 reviews captures 28K/150K businesses
#     b_df = b_df[(b_df["review_count"] > 500)] # high review count businesses for now
    
#     # sampling  businesses
#     sample_size = min(CUTTOFF-len(b_ids), len(b_df))
#     b_ids = np.append(b_ids, b_df["business_id"].sample(sample_size).values)
    
#     if len(b_ids) >= CUTTOFF: # cut off
#         break
    
####### METHOD 1: iterating through businesses first #######
# USERS:      {ID: (friends_ID)} # friends_ID is a set of user ids for constant time lookup
# BUSINESSES: {ID: (Usr_ID)}  # Usr_ID is also a set
    # Assuming we created the data structure above we now preform the following:
    #    Note that this will miss P(User writes a review | 0 friends wrote a review)?
    #        ^ This isn't actually true but I feel like Im still missing something related to this
        # prob_counts = {} # keeps track of instances of (User writes a review | i friend(s) wrote a review) where i is the key
        # for usr_ids in BUSINESSES:
        #     for id in usr_ids:
        #         num_rev = 0
        #         for friend_id in USERS[id]:
        #             if friend_id in usr_ids:
        #                 num_rev += 1
                
        #         # add to the probabilities dictionary or create it
        #         if num_rev in prob_counts:
        #             prob_counts[num_rev] += 1
        #         else:
        #             prob_counts[num_rev] = 1

    # in the worst case everyone is friends with everyone and so this would run in O(b*n^2) 
    # where n is the number of users and b is the number of businesses
    
####### METHOD 2: iterating through users first #######
# USERS:    {ID: {"network": (friends_ID), 
#               "businesses": {business_ID: (review_ID)}
#               }
#           }
# # %% Iterate through the users and create dictionary of user ids and their network
# USERS = {}
# MAX_USERS = None
# ur = qy.get_json_reader(YELP_USER_PATH, chunksize=1000)
# for chunk in tqdm(ur):
#     for usr_id, f_ids, r_c in zip(chunk["user_id"], chunk["friends"], chunk['review_count']):
#         # we are ignoring users with no reviews
#         if r_c > 0:
#             f_ids = set([x.strip() for x in f_ids.split(",")])
#             if len(f_ids) > 0: # we are ignoring users with no friends
#                 USERS[usr_id] = {"network": f_ids, 
#                                  "businesses": {}} # we will fill this in later
                
#                 if MAX_USERS is not None and len(USERS) >= MAX_USERS:
#                     break
#     else:
#         continue
#     break # breaks out of outer loop only if we break out of inner loop

# # %%
# rr = qy.get_json_reader(YELP_REVIEWS_PATH, chunksize=1000)
# # populates USERS with their reviews
# for chunk in tqdm(rr):
#     for usr_id, b_id, r_id in zip(chunk["user_id"], chunk["business_id"], chunk["review_id"]):        
#         if usr_id in USERS:
#             if b_id in USERS[usr_id]["businesses"]:
#                 USERS[usr_id]["businesses"][b_id].add(r_id)
#             else: # if the user has not reviewed this business we add it with the review id
#                 USERS[usr_id]["businesses"][b_id] = {r_id}

rr = qy.get_json_reader(YELP_REVIEWS_PATH, chunksize=1000)
# populates USERS with their reviews (this will miss users with no reviews)
for chunk in tqdm(rr): #reviews has columns: review_id, user_id, business_id, stars, useful, funny, cool, text, date
    for usr_id, bus_id, rev_id in zip(chunk["user_id"], chunk["business_id"], chunk["review_id"]):        
        if usr_id in USERS:
            if bus_id in USERS[usr_id]["businesses"]:
                USERS[usr_id]["businesses"][bus_id].add(rev_id)
            else: # if the user has not reviewed this business we add it with the review id
                USERS[usr_id]["businesses"][bus_id] = {rev_id}
        else: # User has not been seen before
            USERS[usr_id] = {"network": None, # this will be populated when we iterate through the users
                             "businesses": {bus_id: {rev_id}}}
        # limiting number of users
        if MAX_USERS and len(USERS) >= MAX_USERS:
            break
    else: # if the for loop didn't break
        continue
    break

#%% now we iterate through the users and to populate their network
ur = qy.get_json_reader(YELP_USER_PATH, chunksize=1000)
for chunk in tqdm(ur):
    for usr_id, f_ids in zip(chunk["user_id"], chunk["friends"]):
        # again we are ignoring users with no reviews
        if usr_id in USERS:
            f_ids = set([x.strip() for x in f_ids.split(",")])
            
            if len(f_ids) > 0:
                USERS[usr_id]["network"] = f_ids
            else:
                del USERS[usr_id] # removing users with no friends

# %% to get the probabilities we preform the following monte carlo simulation:
prob_counts = {} # keeps track of instances of (User writes a review | i friend(s) wrote a review) where i is the key
for usr_id in tqdm(USERS):
    usr = USERS[usr_id]
    for b in usr["businesses"]:
        num_rev = 0
        for friend_id in usr["network"]:
            # users with no reviews are not in the USERS dictionary
            if friend_id in USERS and b in USERS[friend_id]["businesses"]: # constant time lookup
                num_rev += 1
        
        # add to the probabilities dictionary or create it
        if num_rev in prob_counts:
            prob_counts[num_rev] += 1
        else:
            prob_counts[num_rev] = 1

# Saving the probabilities as a csv with the columns: num_friends, num_instances
with open('pr-user_reviews-num_friends.pkl', 'wb') as f:
    pickle.dump(prob_counts, f)


# %% plotting the probabilities of prob_counts:
prob_sorted = sorted(prob_counts.items())
plt.scatter([x[0] for x in prob_sorted][3:],
            [y[1] for y in prob_sorted][3:])
plt.xlabel("friend counts")
plt.ylabel("Frequency")
plt.show()

# this should also run in O(b*n^2) 
# will use this to validate that i havent missed anything in the first method

# iterate through users and put them into a dictionary with the key being the user id
    # and the value being another dict of business ids that they have reviewed
    # and the value of that dict is the review text
# this allows for constant time look up of the reviews of a user and business

# %%
