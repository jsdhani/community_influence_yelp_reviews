"""
A couple pre-trained sentiment analysis models that I will test:
* Vader
* TextBlob
* Happy Transformer

"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.query_raw_yelp import QueryYelp as qy
from common.config_paths import YELP_DATA
import pandas as pd
import numpy as np

class VADER_SentimentAnalysis:
    def __init__(self):
        self.VADER = SentimentIntensityAnalyzer()

    def get_sentiment(self, text: (str or pd.DataFrame)):
        """
        Gets the sentiment for a given text or dataframe of texts.

        Args:
            text (str or pd.DataFrame): can be a dataframe containing a "text" column or a string.

        Returns:
            pd.Series or pd.Dataframe: series or dataframe containing the sentiment scores for each text.
        """
        if type(text) is str:
            # vader returns a dict of (pos, neg, neu, compound) scores
            return pd.Series(self.VADER.polarity_scores(text))
        else:
            return text['text'].apply(lambda x: pd.Series(self.VADER.polarity_scores(x)))
        

    def get_sentiment_distribution(self, df: pd.DataFrame, bins=20, plot=False):
        """
        Gets the sentiment distribution for reviews in the given dataframe.


        Args:
            df (pd.DataFrame): The dataframe containing the reviews (must have "text" column).
            bins (int, optional): Number of bins to use when counting frequencies. Defaults to 20.
            plot (bool, optional): flag to plot distribution. Defaults to False.

        Returns:
            pd.DataFrame: dataframe containing the scores.
        """
        scores = self.get_sentiment(df)["compound"] # compound score takes into account other socres (pos, neg, neu)
        
        if plot:
            scores.hist(bins=bins)
        return scores
        
    
