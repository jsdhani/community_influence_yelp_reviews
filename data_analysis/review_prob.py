# Gets review probabilities using Monte Carlo Sampling
"""
Determine correlation between the probability a user will review a business to the probability 
it was previously reviewed by another user in their network for T1.
(This gives us an idea of if users typically only go to restaurants that have already been 
visited by their friends)

We use Monte Carlo methods to get the probabilities by:
    1. Iterate through all businesses, 
        counting all the times a user writes a review and 
        the corresponding number of friends that also wrote 
        a review on that same business.
    2. Bin the probabilities so that we have 
        P(User writes a review | i friend(s) wrote a review) 
    3. Where i <= max number reviews written by friends 
        on a single business.
    4. Determine significance of correlation
    5. Conduct correlation and regression analysis on the 
        probability that a user will write a review to the 
        probability their friend wrote a review.

"""

import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from utils.query_raw_yelp import QueryYelp as qy
from common.config_paths import (
    YELP_REVIEWS_PATH,
    YELP_USER_PATH, 
    MT_RESULTS_PATH)

class ReviewProb:
    def __init__(self, max_users=None, chunksize=1000) -> None:
        self.MAX_USERS = max_users
        self.CHUNKSIZE = chunksize
        
        self.prob = None
        self.SAVE_PATH = f'{MT_RESULTS_PATH}pr-user_reviews-num_friends-{self.MAX_USERS}.pkl'
    
    def prep_data(self):
        """
        Prepares the data for performing monte carlo sampling.
        
        2 methods proposed:
        ####### METHOD 1: iterating through businesses first #######
                # USERS:      {ID: (friends_ID)}
                # BUSINESSES: {ID: (Usr_ID)}
        
        ####### METHOD 2: iterating through users first #######
                # USERS:      {ID: {"network": (friends_ID), 
                #                 "businesses": {business_ID: (review_ID)}
                #                 }
                #             }
        
        This method uses method 2.
        """
        # Iterate through the users and create dictionary of user ids and their network
        print("Creating user dictionary...\n\t(<2mins for all users on intel i9-12900k)")
        self.users = {}
        for chunk in tqdm(qy.get_json_reader(YELP_USER_PATH, chunksize=self.CHUNKSIZE)):
            for usr_id, f_ids, r_c in zip(chunk["user_id"], chunk["friends"], chunk['review_count']):
                # We are ignoring users with no reviews
                if r_c > 0:
                    f_ids = set([x.strip() for x in f_ids.split(",")])
                    # we are ignoring users with no friends
                    if len(f_ids) > 0: 
                        self.users[usr_id] = {"network": f_ids, 
                                              "businesses": {}} # populated later
                        
                        if self.MAX_USERS is not None and len(self.users) >= self.MAX_USERS:
                            break
            else:
                continue
            break # breaks out of outer loop only if we break out of inner loop

        # populates USERS with their reviews
        print("Populating user dictionary with reviews...\n\t(<2mins for all reviews on intel i9-12900k)")
        for chunk in tqdm(qy.get_json_reader(YELP_REVIEWS_PATH, chunksize=self.CHUNKSIZE)):
            for usr_id, b_id, r_id in zip(chunk["user_id"], chunk["business_id"], chunk["review_id"]):        
                if usr_id in self.users:
                    if b_id in self.users[usr_id]["businesses"]:
                        self.users[usr_id]["businesses"][b_id].add(r_id)
                    else: # if the user has not reviewed this business we add it with the review id
                        self.users[usr_id]["businesses"][b_id] = {r_id}
        print("Finished prepping data!")
        return self.users
    
    def get_prob(self, plot=True, save=True):
        # To get the probabilities we preform the following monte carlo simulation:
        prob_counts = {} # keeps track of instances of (User writes a review | i friend(s) wrote a review) where i is the key
        for usr_id in tqdm(self.users):
            usr = self.users[usr_id]
            for b in usr["businesses"]:
                num_rev = 0
                for friend_id in usr["network"]:
                    # users with no reviews are not in the USERS dictionary
                    if (friend_id in self.users and 
                        b in self.users[friend_id]["businesses"]): # constant time lookup
                        num_rev += 1
                
                # add to the probabilities dictionary or create it
                if num_rev in prob_counts:
                    prob_counts[num_rev] += 1
                else:
                    prob_counts[num_rev] = 1

        # normalize to get probabilities
        total = sum(prob_counts.values())
        self.prob = {k: v/total for k, v in prob_counts.items()}
        
        if save:
            # Saving the probabilities as a csv with the columns: num_friends, num_instances
            with open(self.SAVE_PATH, 'wb') as f:
                pickle.dump(self.prob, f)
                
        if plot:
            prob_sorted = sorted(self.prob.items())
            plt.scatter([x[0] for x in prob_sorted],
                        [y[1] for y in prob_sorted])
            plt.xlabel("friend counts")
            plt.ylabel("Frequency")
            plt.show()
            
        return self.prob