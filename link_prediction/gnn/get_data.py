import pandas as pd
import os
import pickle
from common.config_paths import NETWORK_DATA_PATH

class GetGNNData:

    def __init__(self):

        with open(
                os.path.dirname(os.path.dirname(os.getcwd())) +
                NETWORK_DATA_PATH + "../" + "pandemic_reviews.pkl", 'rb'
        ) as f:
            self.reviews = pickle.load(f)

        with open(
                os.path.dirname(os.path.dirname(os.getcwd())) +
                NETWORK_DATA_PATH + "pandemic_friends.pkl", 'rb'
        ) as f:
            self.friends = pickle.load(f)

        with open(
                os.path.dirname(os.path.dirname(os.getcwd())) +
                NETWORK_DATA_PATH + "pandemic_users.pkl", 'rb'
        ) as f:
            self.users = pickle.load(f)

        with open(
                os.path.dirname(os.path.dirname(os.getcwd())) +
                NETWORK_DATA_PATH + "pandemic_business.pkl", 'rb'
        ) as f:
            self.business = pickle.load(f)

    def format_friends(self, limit: int = None):
        if limit is not None:
            self.friends = self.friends[
                self.friends['friends_count'] < limit
            ].reset_index(drop=True)

        self.friends = self.friends.explode("friends").reset_index(drop=True)
        self.friends.drop(columns=['friends_count'], inplace=True)
        self.friends.rename(
            columns={'user_id': 'source', 'friends': 'target'},
            inplace=True)

    def format_reviews(self):
        self.reviews = self.reviews.drop(columns=['text', 'date', 'review_id'])
        self.reviews.rename(
            columns={'user_id': 'source', 'business_id': 'target'},
            inplace=True)

    def format_users(self):
        self.users.drop_duplicates(inplace=True)

        self.users = self.users[
            self.users['user_id'].isin(self.all_friends)
        ].reset_index(drop=True)

        self.users = self.users.set_index('user_id')

    def get_all_friends(self):
        self.all_friends = self.friends['source'].tolist()
        self.all_friends.extend(self.friends['target'].tolist())
        self.all_friends = list(set(self.all_friends))

    def format_business(self):
        self.business = self.business[
            self.business['business_id'].isin(self.reviews['target'])
        ].reset_index(drop=True)

        self.business.drop(columns=['name', 'is_open'], inplace=True)
        self.business.state = self.business.state.astype('category').cat.codes
        self.business.postal_code = self.business.postal_code.astype('category').cat.codes

        encoded_business_features = pd.get_dummies(
            self.business['categories'].apply(pd.Series).stack()
        ).sum(level=0)

        self.business.drop(columns=['categories'], inplace=True)
        self.business = pd.concat([self.business, encoded_business_features], axis=1)
        self.business.drop_duplicates(inplace=True)

        self.business = self.business[
            ~self.business['business_id'].isin(self.users.index.tolist())
        ].reset_index(drop=True)

        self.business = self.business.set_index('business_id')

    def main(self):
        print("Formatting friends.....")
        self.format_friends(limit=50)

        print("Formatting reviews.....")
        self.format_reviews()

        print("Getting all friends.....")
        self.get_all_friends()

        print("Formatting users.....")
        self.format_users()

        self.reviews = self.reviews[
            self.reviews['source'].isin(self.all_friends)
        ].reset_index(drop=True)

        print("Formatting businesses.....")
        self.format_business()

        print("Final filtering......")

        self.friends = self.friends[
            self.friends['target'].isin(self.users.index.tolist())
        ].reset_index(drop=True)

        self.reviews = self.reviews[
            self.reviews['target'].isin(self.business.index.tolist())
        ].reset_index(drop=True)

        self.reviews = self.reviews[
            self.reviews['source'].isin(self.users.index.tolist())
        ].reset_index(drop=True)

        self.friends.index += len(self.reviews)

# if __name__ == '__main__':
#     data = GetGNNData()
#     data.main()
#     with open("gnn_friends_10.pkl", 'wb') as f:
#         pickle.dump(data.friends, f)
#     with open("gnn_reviews_10.pkl", 'wb') as f:
#         pickle.dump(data.reviews, f)
#     with open("gnn_users_10.pkl", 'wb') as f:
#         pickle.dump(data.users, f)
#     with open("gnn_business_10.pkl", 'wb') as f:
#         pickle.dump(data.business, f)
#     print("Done!")


