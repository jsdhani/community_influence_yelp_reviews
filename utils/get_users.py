from utils.query_raw_yelp import QueryYelp
import pickle

class GetUsersFromRaw:

    @staticmethod
    def get_users(
            users_till_date_query: str,
            save_filepath: str,
            save_filename: str
    ):
        result = QueryYelp.query_features(
            filename="users.json",
            query=users_till_date_query
        )
        result = result.loc[
            (result['friends'] != '') &
            (result['average_stars'] > 0)
        ].reset_index(drop=True)

        with open(save_filepath+save_filename, 'wb') as f:
            pickle.dump(result, f)

if __name__ == '__main__':
    GetUsersFromRaw.get_users(
        users_till_date_query="`yelping_since` <= '2021-03-01'",
        save_filepath="../data/pandemic/",
        save_filename="pandemic_users.pkl"
    )
