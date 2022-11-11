from common.constants import raw_read_chunk_size
from common.config_paths import YELP_DATA
from pandas import read_json, DataFrame, concat
import logging
import os
logging.basicConfig(level=logging.INFO)

class QueryYelp:

    @staticmethod
    def get_json_reader(
            file: object,
            chunksize: int = raw_read_chunk_size,
    ):
        """
        Return a JSONReader object for the provided file object.
        This JSONReader is iteratively read to get the
        actual JSON contents. The chunk size used here has
        been pre-defined in the constants section.
        :param file: file object for the specific filename
        :param chunksize: number of lines to read at a time (optional)
        :return: JSONReader object
        """

        return read_json(
                    path_or_buf=file,
                    orient="records",
                    lines=True,
                    chunksize=chunksize,
        )

    @staticmethod
    def get_all_features(
            filename: str = None
    ) -> DataFrame:
        """
        Returns all the features present in a JSON file
        without any additional queries. Uses a JSONReader
        object to iterate through the JSON file in chunks
        and create a master pandas dataframe object.
        :param filename: string filename
        :return: corresponding dataframe object for JSON file.
        """
        merged_chunks = []
        try:
            with open(
                    os.path.dirname(os.getcwd()) +
                    YELP_DATA + filename, "r"
            ) as file:
                reader = QueryYelp.get_json_reader(file=file)
                for chunk in reader:
                    merged_chunks.append(chunk)
        except FileNotFoundError:
            logging.exception("The file" + filename +
                              " not found in Yelp!")
        except Exception as excp:
            logging.exception("Exception raised: " + excp)

        return concat(merged_chunks, ignore_index=True)

    @staticmethod
    def query_features(
            filename: str = None,
            drop_features: list = [],
            query: str = None
    ) -> DataFrame:
        """
        Returns the features not explicitly listed
        to be dropped, with the records further
        filtered according to the provided query.
        Uses a JSONReader object to iterate through
        the JSON file in chunks and create a master
         pandas dataframe object.
        :param filename: string filename
        :param drop_features: list of features
        to not be included in the result
        :param query: for filtering the data records
        :return: corresponding dataframe object
        for JSON file.
        """
        merged_chunks = []

        try:
            with open(
                    os.path.dirname(os.getcwd()) +
                    YELP_DATA + filename, "r"
            ) as file:
                reader = QueryYelp.get_json_reader(file=file)
                for chunk in reader:
                    if query is None:
                        reduced_chunk = chunk.drop(
                            columns=drop_features
                        )
                    else:
                        reduced_chunk = chunk.drop(
                            columns=drop_features
                        ).query(query)
                    merged_chunks.append(reduced_chunk)

        except FileNotFoundError:
            logging.exception("The file" + filename +
                              " not found in Yelp!")
        except Exception as excp:
            logging.exception("Exception raised: " + excp)

        return concat(merged_chunks, ignore_index=True)
