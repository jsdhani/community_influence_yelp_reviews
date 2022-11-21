from common.constants import restaurant_categories, FROM, TO,\
    USER_ID, FRIENDS, CATEGORIES, FRIENDS_COUNT
from sklearn.model_selection import train_test_split
from common.config_paths import NETWORK_DATA_PATH
from tqdm import tqdm
import pandas as pd
import pickle
import os

class GetJaccardCoefficient:

    def __init__(self,
                 friends_upper_limit: int = None,
                 friends_lower_limit: int = None):
        """
        Read friends and reviews files, and filter
        out the reviews for  users present in friends data.
        :param friends_upper_limit: If not none, keep all
        the users with friends less than this limit
        :param friends_lower_limit: If not none, keep all
        the users with friends beyond this limit
        """

        with open(os.path.dirname(
                os.path.dirname(os.getcwd())
                ) + NETWORK_DATA_PATH +
                  "pandemic_reviews_cats.pkl", 'rb') as f:

            self.reviews = pickle.load(f)

        with open(
                os.path.dirname(
                        os.path.dirname(os.getcwd())
                ) + NETWORK_DATA_PATH +
                "pandemic_friends.pkl", 'rb') as f:
            self.friends = pickle.load(f)

        self.format_reviews()

        # filtering out friends as per the specified limit

        if friends_upper_limit is not None:
            self.format_friends(limit=friends_upper_limit,
                                is_upper=True)
        else:
            self.format_friends(limit=friends_lower_limit,
                                is_upper=False)

        all_users = self.get_all_users()

        # filtering out the reviews as per the considered
        # users in the friends data

        self.reviews = self.reviews[
            self.reviews[FROM].isin(all_users)
        ].reset_index(drop=True)

    def get_all_users(self):
        """
        Retrieve a list of all users from the
        'FROM' and 'TO' features of the dataset.
        :return: list of users
        """
        all_users = self.friends[FROM].tolist()
        all_users.extend(self.friends[TO].tolist())
        all_users = list(set(all_users))
        return all_users

    def format_friends(self,
                       limit,
                       is_upper: bool = True):
        """
        Formatting the friends dataset as follows:
        1) Filtering out the users as per the specified limit.
        2) The friends of any given user are mentioned in
        a list format. Exploding the list column into
        separate records.
        3) Renaming columns for further usage.
        :param limit: integer limit value
        :param is_upper: to specify whether the limit is
        for upper limit for lower limit
        :return: None, updates the instance member
        of the class.
        """

        if is_upper:
            self.friends = self.friends[
                self.friends[FRIENDS_COUNT] < limit
            ]
        else:
            self.friends = self.friends[
                self.friends[FRIENDS_COUNT] > limit
            ]

        self.friends = self.friends[[USER_ID, FRIENDS]]
        self.friends = self.friends.explode(FRIENDS)\
            .reset_index(drop=True)
        self.friends.rename(
            columns={USER_ID: FROM, FRIENDS: TO},
            inplace=True)

    def format_reviews(self):
        """
        Select a subset of review features
        and rename the columns for further use.
        :return: None, updates the instance member of the class
        """
        self.reviews = self.reviews[[USER_ID, CATEGORIES]]
        self.reviews.rename(
            columns={USER_ID: FROM, CATEGORIES: TO},
            inplace=True)

    def create_link_set(self, df1, df2):
        """
        Merge the records present in reviews and friends dataset
        :param df1: review/friends dataframe object pandas
        :param df2: review/friends dataframe object pandas
        :return: concatenated dataframe object pandas
        """
        return pd.concat([df1, df2], axis=0)\
            .reset_index(drop=True)

    def get_train_test_sets(self):
        """
        Prepare train test sets such that only
        the review links constitute the test set.
        :return: train and test dataframe object pandas
        """
        train, test = train_test_split(
            self.reviews,
            test_size=0.2
        )
        train = self.create_link_set(
            df1=self.friends,
            df2=train
        )
        intersection = self.get_intersection_values(test, train)

        test = test[
            test[FROM].isin(intersection)
        ].reset_index(drop=True)

        test = test[
            test[TO].isin(intersection)
        ].reset_index(drop=True)

        test = list(test.itertuples(
            index=False,
            name=None))

        return train, test

    def get_intersection_values(self, test, train):
        """
        To prepare an intersection set of users and
        businesses to ensure that all the considered
        values are both in train and test sets
        :param test: dataframe object pandas
        :param train: dataframe object pandas
        :return: list of users/businesses
        """
        test_values = test[FROM].tolist()
        test_values.extend(test[TO].tolist())
        train_values = train[FROM].tolist()
        train_values.extend(train[TO].tolist())
        return list(set(train_values).intersection(set(test_values)))

    def get_neighbors(self, df, value):
        """
        Get all the users/businesses connected
        to a given user
        :param df: dataframe object pandas
        :param value: userID
        :return: set of neighbors
        """
        value_df = df.loc[(df[FROM] == value) |
                          (df[TO] == value)]\
            .reset_index(drop=True)

        if len(value_df) == 0:
            return {}

        neighbors = value_df[FROM].tolist()
        neighbors.extend(value_df[TO].tolist())
        neighbors.remove(value)
        return set(neighbors)

    def get_train_subset(self, train, test_user):
        """
        Get the subset of users belonging to the
        community of the test user
        :param train: dataframe object pandas
        :param test_user: user for whom the
        community is to be filtered
        :return: dataframe object pandas
        """
        neighbors = train.loc[(train[FROM] == test_user) |
                              (train[TO] == test_user)]\
            .reset_index(drop=True)

        neighbor_users = neighbors[FROM].tolist()
        neighbor_users.extend(neighbors[TO].tolist())
        neighbor_users = list(
            set(
                [user for user in neighbor_users
                 if user not in restaurant_categories]
            ))
        return train.loc[(train[FROM].isin(neighbor_users)) |
                         (train[TO].isin(neighbor_users))]\
            .reset_index(drop=True)

    def main(self):
        """
        Driver function for generating Jaccard Coefficient
        :return: None, prints the Average and Maximum JC values
        """
        train, test = self.get_train_test_sets()
        result = []
        print("Computing JCs...")
        for index in tqdm(range(len(test))):
            record = test[index]
            test_user = record[0]
            test_cat = record[1]
            train_subset = self.get_train_subset(train=train, test_user=test_user)
            user_neighbors = self.get_neighbors(df=train_subset, value=test_user)
            cat_neighbors = self.get_neighbors(df=train_subset, value=test_cat)
            result.append(
                len(user_neighbors.intersection(cat_neighbors)) /
                len(user_neighbors.union(cat_neighbors)
                    ))
        print("Average JC = ", sum(result)/len(result))
        print("Max JC = ", max(result))


if __name__ == '__main__':
    LOWER_LIMIT = 5000
    UPPER_LIMIT = 3
    restaurant_categories = list(
        map(
            lambda x: x.lower(),
            restaurant_categories
        )
    )
    g = GetJaccardCoefficient(friends_upper_limit=UPPER_LIMIT)
    g.main()
    g2 = GetJaccardCoefficient(friends_lower_limit=LOWER_LIMIT)
    g2.main()
