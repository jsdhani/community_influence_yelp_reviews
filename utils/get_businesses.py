from common.constants import restaurant_categories
from utils.query_raw_yelp import QueryYelp
import pickle

class GetBusinessesFromRaw:

    @staticmethod
    def filter_categories(result):
        result['categories'] = [cat.lower().split(",") if cat is not None
                                else "" for cat in result['categories']]
        result['categories'] = [list(map(lambda x: x.strip(), categories))
                                for categories in result['categories']]
        result['categories'] = [list(
            filter(
                lambda x: x in restaurant_categories,
                categories
            )
        )
            for categories in result['categories']]
        result = result[
            result['categories'].map(lambda cat: len(cat)) > 0
            ].reset_index(drop=True)
        return result

    @staticmethod
    def get_businesses(
            save_filepath: str,
            save_filename: str,
            drop_features: list = []
    ):
        result = QueryYelp.query_features(
            filename="business.json",
            drop_features=drop_features)

        result = GetBusinessesFromRaw.filter_categories(result)

        with open(save_filepath+save_filename, 'wb') as f:
            pickle.dump(result, f)

if __name__ == '__main__':

    restaurant_categories = list(map(
        lambda x: x.lower(),
        restaurant_categories))

    GetBusinessesFromRaw.get_businesses(
        save_filepath="../data/pandemic/",
        save_filename="pandemic_businesses.pkl",
        drop_features=['attributes']
    )