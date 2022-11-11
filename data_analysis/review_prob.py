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
import pandas as pd
import numpy as np

from data_analysis.sentiment_models import SentimentAnalysis
from utils.query_raw_yelp import QueryYelp as qy
from common.config_paths import (
    YELP_REVIEWS_PATH, 
    YELP_BUSINESS_PATH, 
    YELP_USER_PATH)

class ReviewProb:
    def __init__(self, df: pd.DataFrame, model: SentimentAnalysis):
        self.df = df
        self.model = model
        self.scores = None
        self.lengths = None
        self.prob = None
        
    def prep_data(self, max_users=None, chunksize=1000):
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
        # Iterate through the reviews and create dictionary of user ids and their network
        rr = qy.get_json_reader(YELP_REVIEWS_PATH, chunksize=chunksize)
        self.users = {}
        MAX_USERS = max_users
        # Populates USERS with all the reviews (this will miss users with no reviews)
        for chunk in tqdm(rr):
            for u_id, b_id, r_id in zip(chunk["user_id"], chunk["business_id"], chunk["review_id"]):        
                if u_id in self.users:
                    if b_id in self.users[u_id]["businesses"]:
                        self.users[u_id]["businesses"][b_id].add(r_id)
                    else: # if the user has not reviewed this business we add it with the review id
                        self.users[u_id]["businesses"][b_id] = {r_id}
                else: # User has not been seen before
                    self.users[u_id] = {"network": None, # this will be populated when we iterate through the users
                                    "businesses": {b_id: {r_id}}}
                # limiting number of users
                if MAX_USERS and len(self.users) >= MAX_USERS:
                    break
            else: # if the for loop didn't break
                continue
            break

        # now we iterate through the users and to populate their network
        ur = qy.get_json_reader(YELP_USER_PATH, chunksize=chunksize)
        for chunk in tqdm(ur):
            for u_id, f_ids in zip(chunk["user_id"], chunk["friends"]):
                # again we are ignoring users with no reviews
                if u_id in self.users:
                    f_ids = set([x.strip() for x in f_ids.split(",")])
                    
                    if len(f_ids) > 0:
                        self.users[u_id]["network"] = f_ids
                    else:
                        del self.users[u_id] # removing users with no friends

    def get_prob(self, n_samples=1000, bins=100, plot=False):
        """
        Gets the probability of a review being positive given its length.

        Args:
            n_samples (int, optional): Number of samples to take. Defaults to 1000.
            bins (int, optional): Number of bins to use when counting frequencies. Defaults to 100.
            plot (bool, optional): flag to plot distribution. Defaults to False.

        Returns:
            (pd.Series, Tuple(np.array,np.array)): tuple of the review lengths and their distribution (review lengths, (freq, bins))
        """
        # Get the sentiment scores for the reviews
        self.scores = self.model.get_sentiment(self.df)
        # Get the review lengths
        self.lengths = self.df["text"].apply(lambda x: len(x))
        # Get the probability of a review being positive given its length
        self.prob = self._get_prob(self.scores, self.lengths, n_samples, bins)
        if plot:
            self._plot_prob(self.prob)
        return self.prob 