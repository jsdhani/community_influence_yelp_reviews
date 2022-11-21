import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np

from utils.query_raw_yelp import QueryYelp as qy
from common.config_paths import (
    YELP_REVIEWS_PATH,
    YELP_USER_PATH, 
    RATINGS_CORR_PATH,
    YELP_BUSINESS_PATH)
from common.constants import restaurant_categories

class RatingAnalysis:
    def __init__(self, chunksize=1000, save_path=RATINGS_CORR_PATH):
        self.CHUNKSIZE = chunksize
        
        self.SAVE_PATH_FN = lambda x: f"{save_path}ratings_{x}.pkl"
        self.SAVE_PATH = None
        
class RatingAnalysisGeneral(RatingAnalysis):
    def __init__(self, chunksize=1000, save_path=RATINGS_CORR_PATH):
        super().__init__(chunksize, save_path)    
        self.bus_ratings = {}
    
    def get_business_rating(self, date_range=(pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01')), 
                    filter=restaurant_categories, exclude_filter=False):
        """
        Creates dictionary:
        Bus_reviews = {"bus_id": average_rating, ...}
        
        """
        self.SAVE_PATH = self.SAVE_PATH_FN(
                f"{date_range[0].strftime('%Y-%m-%d')}_{date_range[1].strftime('%Y-%m-%d')}")
        
        
        # looping through businesses to determine if they fall within filter
        b_cleared = set() # added if they are cleared
        if filter:
            for chunk in tqdm(qy.get_json_reader(YELP_BUSINESS_PATH, chunksize=self.CHUNKSIZE),
                            desc="Applying business filter"):
                for bus_id, ctgs in zip(chunk['business_id'], chunk['categories']):
                    if (ctgs == None) or (bus_id in b_cleared): continue # skipping if already checked
                    
                    for c in ctgs.split(','):
                        filt_bool = c.strip() in filter
                        
                        # filter data tells us what to exclude if this flag is set:
                        if exclude_filter: filt_bool = not filt_bool 
                        
                        if filt_bool:
                            b_cleared.add(bus_id)
                            break
        
        # traverse through the reviews to get:
        #       ratings_num = {b_id: # of ratings}
        #       ratings_totV = {b_id: ratingtot}
        rating_num = {}
        rating_totV = {}
        for chunk in tqdm(qy.get_json_reader(YELP_REVIEWS_PATH, chunksize=self.CHUNKSIZE), 
                                                desc="Getting number of ratings and total value per business"):
            for bus_id, rating, date in zip(chunk['business_id'], chunk['stars'], chunk['date']):
                
                if filter and (bus_id not in b_cleared): continue # filtering out b_id by catagory if provided
                    
                # filtering out by date range
                if date_range[0] <= date <= date_range[1]:
                    if bus_id not in rating_num: # add business_id to dictionary if not already present
                        rating_num[bus_id] = 1
                        rating_totV[bus_id] = int(rating)
                    else:
                        rating_num[bus_id] += 1
                        rating_totV[bus_id] += int(rating)
                    
        # now we do ratings_totV/ratings_num to get the average rating for each business
        self.bus_ratings = {b_id: rating_totV[b_id]/rating_num[b_id] for b_id in tqdm(rating_num, desc="Calculating averages")}
        
        return self.bus_ratings
    
    def save_ratings(self):
        with open(self.SAVE_PATH, 'wb') as f:
            pickle.dump(self.bus_ratings, f)

class RatingAnalysisFriends(RatingAnalysis):
    def __init__(self, chunksize=1000, save_path=RATINGS_CORR_PATH):
        super().__init__(chunksize, save_path)
        
        self.bus_revs = {} # {business_id:{user_id:rating, ...}, ...}
        self.network = {} # {user_id:{friend_id, ...}, ...}
        self.ratings = None
        
        
    def prep_data(self, date_range=(pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01')), 
                  filter=restaurant_categories, exclude_filter=False):
        """
        Prepares the data for analysis by filtering out reviews that are not within the date range
        and removing any reviews that have a rating of 0.
        
        2 directories:
        friends =       {
                            user_id:{friend_id, ...}, # set for O(1) lookup
                            ...
                        }
        bus_reviews =   {
                            business_id:{
                                user_id:rating,
                                ...
                                }, 
                            ...
                        }
                        
                        {
        """
        self.SAVE_PATH = self.SAVE_PATH_FN(
                f"{date_range[0].strftime('%Y-%m-%d')}_{date_range[1].strftime('%Y-%m-%d')}")
        
        
        # looping through businesses to determine if they fall within filter
        b_cleared = set() # added if they are cleared
        if filter:
            for chunk in tqdm(qy.get_json_reader(YELP_BUSINESS_PATH, chunksize=self.CHUNKSIZE),
                            desc="Applying business filter"):
                for bus_id, ctgs in zip(chunk['business_id'], chunk['categories']):
                    if (ctgs == None) or (bus_id in b_cleared): continue # skipping if already checked
                    
                    for c in ctgs.split(','):
                        filt_bool = c.strip() in filter
                        
                        # filter data tells us what to exclude if this flag is set:
                        if exclude_filter: filt_bool = not filt_bool 
                        
                        if filt_bool:
                            b_cleared.add(bus_id)
                            break
        
        # traverse through the reviews to get business_id and user_id for ratings
        # also filter out reviews that are not within the date range
        # this will create a dictionary of {business_id:{user_id:rating, ...}, ...}
        for chunk in tqdm(qy.get_json_reader(YELP_REVIEWS_PATH, chunksize=self.CHUNKSIZE), 
                                                desc="Creating bus_revs dictionary"):
            for usr_id, bus_id, rating, date in zip(chunk['user_id'], chunk['business_id'], 
                                                    chunk['stars'], chunk['date']):                     
                
                if filter and (bus_id not in b_cleared): continue # filtering out b_id by catagory if provided
                    
                # filtering out by date range
                if date_range[0] <= date <= date_range[1]:
                    if bus_id not in self.bus_revs: # add business_id to dictionary if not already present
                        self.bus_revs[bus_id] = {}
                    # adding user rating to that business
                    self.bus_revs[bus_id][usr_id] = rating
                    
        # traverse through the users to get friends
        for chunk in tqdm(qy.get_json_reader(YELP_USER_PATH, chunksize=self.CHUNKSIZE), 
                                                desc="Creating network dictionary"):
            for usr_id, f_ids in zip(chunk['user_id'], chunk['friends']):
                assert usr_id not in self.network, "User ID already in network dictionary"
                self.network[usr_id] = set([x.strip() for x in f_ids.split(",")])
    
    def get_ratings(self):
        """
        iterates through the bus_revs dictionary to find the ratings of the user and friends
        then pairs the users rating with the average friend rating for each business.
        
        the final output is a np.array in the form [(user_rating, friend_rating)] where each 
        instance is a business-friend group
        """
        ratings = []
        for bus_id, reviews in tqdm(self.bus_revs.items(), desc="Getting ratings"):
            for usr_id, rating in reviews.items():
                # checking to see if the user is in the network
                if usr_id in self.network:
                    # looping through their friends to get their ratings and average them
                    avg_friend_rating = 0
                    num_friends_rated = 0
                    for friend_id in self.network[usr_id]:
                        if friend_id in reviews:
                            avg_friend_rating += reviews[friend_id]
                            num_friends_rated += 1
                            
                    if num_friends_rated > 0: # only add to ratings if there are friends that rated the business
                        ratings.append((rating, avg_friend_rating/num_friends_rated))
                    
        self.ratings = np.array(ratings) # convert to numpy array for easier manipulation
        return self.ratings
    
    def save_ratings(self):
        with open(self.SAVE_PATH, 'wb') as f:
            pickle.dump(self.ratings, f)
            
    def plot_ratings(self, title="Ratings Correlation"):
        plt.scatter(self.ratings[:,0], self.ratings[:,1])
        plt.xlabel("User Rating")
        plt.ylabel("Average Friend Rating")
        plt.title(title)
        plt.show()