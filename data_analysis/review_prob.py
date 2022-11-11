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

from data_analysis.sentiment_models import SentimentAnalysis
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ReviewProb:
    def __init__(self, df: pd.DataFrame, model: SentimentAnalysis):
        self.df = df
        self.model = model
        self.scores = None
        self.lengths = None
        self.prob = None

    def get_prob(self, n_samples=1000, bins=100, plot=False):
        """
        Gets the probability of a review being positive given its length.

        Args:
            n_samples (int, optional): Number of samples to take. Defaults to 1000.
            bins (int, optional): Number of bins to use when counting frequencies. Defaults to 100.
            plot (bool, optional): flag to plot distribution. Defaults to False.

        Returns:
            (pd.Series, Tuple(np.array,np.array)): tuple of the review lengths and their distribution (review lengths, (freq, bins))
        """
        # Get the sentiment scores for the reviews
        self.scores = self.model.get_sentiment(self.df)
        # Get the review lengths
        self.lengths = self.df["text"].apply(lambda x: len(x))
        # Get the probability of a review being positive given its length
        self.prob = self._get_prob(self.scores, self.lengths, n_samples, bins)
        if plot:
            self._plot_prob(self.prob)
        return self.prob 