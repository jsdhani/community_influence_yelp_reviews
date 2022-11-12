from utils.query_raw_yelp import QueryYelp
import pickle

class GetReviewsFromRaw:

    @staticmethod
    def get_reviews(
            date_from_query: str,
            date_to_query: str,
            save_filepath: str,
            save_filename: str
    ):
        result = QueryYelp.query_features(
            filename="reviews.json",
            query=date_from_query
        )
        result = result.query(date_to_query).reset_index(drop=True)
        with open(save_filepath+save_filename, 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    GetReviewsFromRaw.get_reviews(
        date_from_query="`date` >= '2019-12-01'",
        date_to_query='date <= "2021-08-01 00:00:00"',
        save_filepath='../data/pandemic/',
        save_filename='pandemic_reviews.pkl'
    )
