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

rp = ReviewProb()
rp.prep_data_range(date_range=(pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01')))
rp.get_prob(plot=True, save=True)





























# # %%
# rr = qy.get_json_reader(YELP_REVIEWS_PATH, chunksize=1000)
# # we start with the reviews to filter specific time periods
# USERS = {}
# MAX_USERS = 10000
# date_range = (pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01'))
# for chunk in tqdm(rr):
#     for usr_id, bus_id, rev_id, date in zip(
#                     chunk["user_id"], chunk["business_id"], 
#                     chunk["review_id"], chunk["date"]):     
        
#         # ensuring that we only get reviews from 2019-12-01 to 2021-08-01 (YYYY-MM-DD)
#         if date >= date_range[0] and date <= date_range[1]:
#             if usr_id not in USERS:
#                 USERS[usr_id] = {"businesses": {bus_id: [rev_id]}} # network is added later
#             else:
#                 if bus_id not in USERS[usr_id]["businesses"]:
#                     USERS[usr_id]["businesses"][bus_id] = [rev_id]
#                 else:
#                     USERS[usr_id]["businesses"][bus_id].append(rev_id)
    
#             # limiting number of users for space constraints
#             if MAX_USERS and len(USERS) >= MAX_USERS:
#                 break
#     else: # if the for loop didn't break
#         continue
#     break

# #%% now we iterate through the users and to populate their network
# ur = qy.get_json_reader(YELP_USER_PATH, chunksize=1000)
# for chunk in tqdm(ur):
#     for usr_id, f_ids in zip(chunk["user_id"], chunk["friends"]):
#         # again we are ignoring users with no reviews and users with no friends in our time period
#         if usr_id in USERS and f_ids != "None":
#             f_ids = set([x.strip() for x in f_ids.split(",")])
            
#             if len(f_ids) > 0:
#                 USERS[usr_id]["network"] = f_ids
#             else:
#                 del USERS[usr_id] # removing users with no friends

# # %% to get the probabilities we preform the following monte carlo simulation:
# prob_counts = {} # keeps track of instances of (User writes a review | i friend(s) wrote a review) where i is the key
# for usr_id in tqdm(USERS):
#     usr = USERS[usr_id]
#     for b in usr["businesses"]:
#         num_rev = 0
#         for friend_id in usr["network"]:
#             # users with no reviews are not in the USERS dictionary
#             if friend_id in USERS and b in USERS[friend_id]["businesses"]: # constant time lookup
#                 num_rev += 1
        
#         # add to the probabilities dictionary or create it
#         if num_rev in prob_counts:
#             prob_counts[num_rev] += 1
#         else:
#             prob_counts[num_rev] = 1

# # Saving the probabilities as a csv with the columns: num_friends, num_instances
# with open('pr-user_reviews-num_friends.pkl', 'wb') as f:
#     pickle.dump(prob_counts, f)


# # %% plotting the probabilities of prob_counts:
# prob_sorted = sorted(prob_counts.items())
# plt.scatter([x[0] for x in prob_sorted][3:],
#             [y[1] for y in prob_sorted][3:])
# plt.xlabel("friend counts")
# plt.ylabel("Frequency")
# plt.show()
# %%
