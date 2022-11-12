from utils.query_raw_yelp import QueryYelp
import pickle

class GetBusinessesFromRaw:

    @staticmethod
    def get_businesses(
            save_filepath: str,
            save_filename: str,
            drop_features: list = []
    ):
        result = QueryYelp.query_features(
            filename="business.json",
            drop_features=drop_features)
        with open(save_filepath+save_filename, 'wb') as f:
            pickle.dump(result, f)

if __name__ == '__main__':
    GetBusinessesFromRaw.get_businesses(
        save_filepath="../data/pandemic/",
        save_filename="pandemic_businesses.pkl",
        drop_features=['attributes']
    )