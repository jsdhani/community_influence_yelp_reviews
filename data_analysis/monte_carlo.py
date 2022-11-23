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
import pandas as pd

from utils.query_raw_yelp import QueryYelp as qy
from common.config_paths import (
    YELP_REVIEWS_PATH,
    YELP_USER_PATH, 
    MC_RESULTS_PATH)

class ReviewProb:
    def __init__(self, max_users=None, chunksize=1000, save_path=MC_RESULTS_PATH) -> None:
        self.MAX_USERS = max_users
        self.CHUNKSIZE = chunksize
        
        self.prob_0 = {}
        self.prob_1 = {}
        self.users = {}
        self.SAVE_PATH_FN = lambda x: f"{save_path}{x}.pkl" # x is a placeholder for the date range
        self.SAVE_PATH = self.SAVE_PATH_FN("all") # default is entire date range
    
    def prep_data_range(self, date_range=(pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01'))):
        """
        Prepares the data for performing monte carlo sampling of reviews within a certain period of time.

        Args:
            date_range (tuple, optional): time periods in pd.Timestamp form. 
                Defaults to ('2019-12-01', '2021-08-01').
        """
        # chaning the save path to reflect the date range
        self.SAVE_PATH = self.SAVE_PATH_FN(
                f"{date_range[0].strftime('%Y-%m-%d')}_{date_range[1].strftime('%Y-%m-%d')}")
        
        self.all_businesses = set() # used to get P(0|0) later
        
        # we start with the reviews to filter specific time periods
        for chunk in tqdm(qy.get_json_reader(YELP_REVIEWS_PATH, chunksize=self.CHUNKSIZE),
                          desc="Creating user dictionary"):
            for usr_id, bus_id, rev_id, date in zip(
                            chunk["user_id"], chunk["business_id"], 
                            chunk["review_id"], chunk["date"]):     
                
                # ensuring that we only get reviews from specific time range (YYYY-MM-DD)
                if date >= date_range[0] and date <= date_range[1]:
                    # adding to all businesses reviewed in the time period
                    if bus_id not in self.all_businesses: self.all_businesses.add(bus_id)
                    
                    if usr_id not in self.users:
                        self.users[usr_id] = {"businesses": {bus_id: [rev_id]}} # network is added later
                    else:
                        if bus_id not in self.users[usr_id]["businesses"]:
                            self.users[usr_id]["businesses"][bus_id] = [rev_id]
                        else:
                            self.users[usr_id]["businesses"][bus_id].append(rev_id)
            
                    # limiting number of users for space constraints
                    if self.MAX_USERS and len(self.users) >= self.MAX_USERS:
                        break
            else: # if the for loop didn't break
                continue
            break

        # now we iterate through the users and to populate their network
        for chunk in tqdm(qy.get_json_reader(YELP_USER_PATH, chunksize=self.CHUNKSIZE), 
                          desc="Populating user dictionary with reviews"):
            for usr_id, f_ids in zip(chunk["user_id"], chunk["friends"]):
                # again we are ignoring users with no reviews and users with no friends in our time period
                if usr_id in self.users and f_ids != "None":
                    f_ids = set([x.strip() for x in f_ids.split(",")])
                    
                    if len(f_ids) > 0:
                        self.users[usr_id]["network"] = f_ids
                    else:
                        del self.users[usr_id] # removing users with no friends
                        
        # final check to remove users with no reviews or no friends
        no_network = [k for k,v in self.users.items() if "network" not in v]
        for k in tqdm(no_network, desc="Final Check: Removing users with no friends"):
            del self.users[k]
            
        print("Finished prepping data!")
        return self.users
    
    def prep_data(self):
        """
        Prepares the data for performing monte carlo sampling.
        
        2 methods proposed:
        ####### METHOD 1: iterating through businesses first #######
                # USERS:      {ID: (friends_ID)}
                # BUSINESSES: {ID: (Usr_ID)}
        
        ####### METHOD 2.1: iterating through users first ####### 
            # Exclude users with no friends or no reviews,
                # USERS:      {ID: {"network": (friends_ID), 
                #                 "businesses": {business_ID: (review_ID)}
                #                 }
                #             }               
        
        This method uses method 2
        """
        # Iterate through the users and create dictionary of user ids and their network
        print("Creating user dictionary...\n\t(<2mins for all users on intel i9-12900k)")
        for chunk in tqdm(qy.get_json_reader(YELP_USER_PATH, chunksize=self.CHUNKSIZE)):
            for usr_id, f_ids, r_c in zip(chunk["user_id"], chunk["friends"], chunk['review_count']):
                # We are ignoring users with no reviews OR no friends
                if not (r_c == 0 or f_ids == "None"):
                    f_ids = set([x.strip() for x in f_ids.split(",")])
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
                    else: # if we have not seen this user review this business before we add it with the review id
                        self.users[usr_id]["businesses"][b_id] = {r_id}
        
        # final check to remove users with no reviews or no friends
        print("Finished prepping data! Performing final check...")
        no_network = [k for k,v in self.users.items() if "network" not in v]
        for k in no_network:
            del self.users[k]
                
        return self.users
    
    def get_probs(self, plot=True, save=True, weighted=False, normalize=True):
        """
        To get prob(0|i)s we:
            1. iterate USERS:
                keep track of {b_id: i} # b_id = business that this user did not review but i friends did
                i) iterate friends
                    a - count # of businesses they review that the user doesn't.
                        Add to local var mentioned above
                ii) Get values and add them to prob counts as {i: #evnts}
            2. normalize counts
            
        To get prob(1|i)s we adjust above 1.i and 1.ii to:
            keep track of b_reviewed = {b_id: i} # business that this user did review and i friends did
            i) iterate friends
                a - count # of businesses they review that the user also does.

        Args:
            plot (bool, optional): _description_. Defaults to True.
            save (bool, optional): _description_. Defaults to True.
            weighted (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # To get the probabilities we preform the following monte carlo simulation:
        prob_counts_0 = {} # user has not reviewed business but i friends have
        prob_counts_1 = {} # user has reviewed business and i friends have
        for usr_id in tqdm(self.users, desc="Getting prob(0|i) and prob(1|i)"):
            usr = self.users[usr_id]
            num_f = len(usr["network"])
            b_not_reviewed = {} # business that this user did not review but i friends did (i=value and b_id=key)
            b_reviewed = {}
            
            # looping through friends of user
            for friend_id in usr["network"]:
                if friend_id not in self.users: # friend might have been filtered out (e.g.: no reviews during time period)
                    continue
                friend = self.users[friend_id]
                # looping through businesses reviewed by friend
                for b in friend["businesses"]:
                    # P(1|i) - if user has also reviewed this business
                    if b in usr["businesses"]: # constant time lookup
                        if b in b_reviewed:
                            b_reviewed[b] += 1
                        else:
                            b_reviewed[b] = 1
                    # P(0|i) - user has NOT reviewed this business
                    else: 
                        if b in b_not_reviewed:
                            b_not_reviewed[b] += 1
                        else:
                            b_not_reviewed[b] = 1
            
            # all other businesses that this user did not review or friends didnt review=> P(0|0)
            if 0 not in prob_counts_0: prob_counts_0[0] = 0 # initialize if not already
            prob_counts_0[0] += len(self.all_businesses) - (
                                len(b_not_reviewed) +   # User didnt review but friends did
                                len(b_reviewed))        # User reviewed and friends did (disjoint from b_not_reviewed)
                                                        # = total businesses reviewed by Users and friends
            
            # remaining businesses that this user reviewed but friends didnt => P(1|0)
            for b in usr["businesses"]:
                if b not in b_reviewed: b_reviewed[b] = 0
                                
            # updating prob_counts based on values from b_not_reviewed and b_reviewed:
            for b in b_not_reviewed:
                i = b_not_reviewed[b]
                if i in prob_counts_0:
                    prob_counts_0[i] += num_f if weighted else 1
                else:
                    prob_counts_0[i] = num_f if weighted else 1
            for b in b_reviewed:
                i = b_reviewed[b]
                if i in prob_counts_1:
                    prob_counts_1[i] += num_f if weighted else 1
                else:
                    prob_counts_1[i] = num_f if weighted else 1
            
        
        # normalize to get probabilities
        if normalize:
            prob_counts_0 = {k: v/sum(prob_counts_0.values()) for k,v in prob_counts_0.items()}
            prob_counts_1 = {k: v/sum(prob_counts_1.values()) for k,v in prob_counts_1.items()}
        
        self.prob_0 = prob_counts_0
        self.prob_1 = prob_counts_1
        
        if save:
            f_name = '/'.join(self.SAVE_PATH.split("/")[:-1]) +'/W-' \
                            + self.SAVE_PATH.split("/")[-1] if weighted else self.SAVE_PATH
            
            f_name_0 = f_name.split(".")[0] + "_prob_0.pkl"
            f_name_1 = f_name.split(".")[0] + "_prob_1.pkl"
            # Saving the probabilities as a dict pkl with the columns: num_friends, num_instances
            with open(f_name_0, 'wb') as f:
                pickle.dump(self.prob_0, f)
                
            with open(f_name_1, 'wb') as f:
                pickle.dump(self.prob_1, f)
                
        if plot:
            prob_sorted = sorted(self.prob_0.items())
            plt.scatter([x[0] for x in prob_sorted],
                        [y[1] for y in prob_sorted])
            plt.xlabel("friend counts")
            plt.ylabel("Frequency")
            plt.title("Prob(0|i) - User has not reviewed business but i friends have")
            plt.show()
            
            prob_sorted = sorted(self.prob_1.items())
            plt.scatter([x[0] for x in prob_sorted],
                        [y[1] for y in prob_sorted])
            plt.xlabel("friend counts")
            plt.ylabel("Frequency")
            plt.title("Prob(1|i) - User has reviewed business and i friends have")
            plt.show()
            
            
        return self.prob_0, self.prob_1
    
    def get_prob_1(self, plot=True, save=True, weighted=False):        
        """
        To get prob(1|i)s we:
            1. iterate through USERS
                i) iterate through their reviews
                    a - Count i friends that have also reviewed that same business
                ii) add as {i: # of events} where # events is how often this exact instance occurs
            2. normalize counts
        """
        # To get the probabilities we preform the following monte carlo simulation:
        prob_counts = {} # keeps track of instances of (User writes a review | i friend(s) wrote a review) where i is the key
        for usr_id in tqdm(self.users):
            usr = self.users[usr_id]
            num_f = len(usr["network"])
            # looping through businesses reviewed by user
            for b in usr["businesses"]: 
                num_rev = 0
                # looping through friends of user
                for friend_id in usr["network"]:
                    # users with no reviews are not in the USERS dictionary (should not be an issue as we removed them)
                    if (friend_id in self.users and 
                        b in self.users[friend_id]["businesses"]): # constant time lookup
                        num_rev += 1 # countin number of friends who reviewed the business
                
                # add to the probabilities dictionary or create it
                if num_rev in prob_counts:
                    prob_counts[num_rev] += num_f if weighted else 1
                else:
                    prob_counts[num_rev] = num_f if weighted else 1

        # normalize to get probabilities
        total = sum(prob_counts.values())
        self.prob_1 = {k: v/total for k, v in prob_counts.items()}
        
        if save:
            f_name = '/'.join(self.SAVE_PATH.split("/")[:-1]) +'/W-' \
                            + self.SAVE_PATH.split("/")[-1] if weighted else self.SAVE_PATH
            # Saving the probabilities as a dict pkl with the columns: num_friends, num_instances
            with open(f_name, 'wb') as f:
                pickle.dump(self.prob_1, f)
                
        if plot:
            prob_sorted = sorted(self.prob_1.items())
            plt.scatter([x[0] for x in prob_sorted],
                        [y[1] for y in prob_sorted])
            plt.xlabel("friend counts")
            plt.ylabel("Frequency")
            plt.show()
            
        return self.prob_1
    
    
    
if __name__ == "__main__":
    # prob before covid
    rp = ReviewProb()
    rp.prep_data_range(date_range=(pd.Timestamp('2017-04-01'), pd.Timestamp('2018-12-01')))
    rp.get_prob_1(plot=False, save=True, weighted=True)
    
    # prob after covid
    rp = ReviewProb()
    rp.prep_data_range(date_range=(pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01')))
    rp.get_prob_1(plot=False, save=True)