import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import pandas as pd

from utils.query_raw_yelp import QueryYelp as qy
from common.config_paths import (
    YELP_REVIEWS_PATH,
    YELP_USER_PATH, 
    MC_RESULTS_PATH)

class RatingAnalysis:
    def __init__(self, save_path=):
        
    
    def prep_data(self, date_range=(pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01'))):
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
        
        """
        
        # traverse through the reviews to get business_id and user_id for ratings
        # also filter out reviews that are not within the date range
        # this will create a dictionary of {business_id:{user_id:rating, ...}, ...}
        
        
        # traverse through the users to get friends
        
        return self.data