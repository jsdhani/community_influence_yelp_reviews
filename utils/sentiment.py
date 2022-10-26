"""
A couple pre-trained sentiment analysis models that I will test:
* Vader
* TextBlob - No paper for this one, but it is a very popular library
* Happy Transformer
"""
from tokenize import Number
from unittest import result
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from happytransformer import HappyTextClassification
from happytransformer.happy_text_classification import TextClassificationResult

from textblob import TextBlob

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class SentimentAnalysis(ABC):
    @abstractmethod   
    def get_sentiment(self, text:(str or pd.DataFrame)):
        """
        Gets the sentiment for a given text or dataframe of texts.

        Args:
            text (str or pd.DataFrame): can be a dataframe containing a "text" column or a string.

        Returns:
            pd.Series or pd.Dataframe: series or dataframe containing the sentiment scores for each text.
        """
        pass
    
    @abstractmethod
    def get_sentiment_distribution(self, df: pd.DataFrame, bins=100, plot=False, score=None):
        """
        Gets the sentiment distribution for reviews in the given dataframe.

        Args:
            df (pd.DataFrame): The dataframe containing the reviews (must have "text" column).
            bins (int, optional): Number of bins to use when counting frequencies. Defaults to 20.
            plot (bool, optional): flag to plot distribution. Defaults to False.
            score (str, optional): the sentiment score to use. Defaults to None.

        Returns:
            (pd.Series, Tuple(np.array,np.array)): tuple of the scores and their distribution (scores, (freq, bins))
        """
        pass
    
    @staticmethod
    def get_review_length_distribution(df: pd.DataFrame, bins=100, plot=False):
        """
        Gets the distribution of review lengths for the given dataframe.

        Args:
            df (pd.DataFrame): The dataframe containing the reviews (must have "text" column).
            bins (int, optional): Number of bins to use when counting frequencies. Defaults to 20.
            plot (bool, optional): flag to plot distribution. Defaults to False.

        Returns:
            (pd.Series, Tuple(np.array,np.array)): tuple of the review lengths and their distribution (review lengths, (freq, bins))
        """
        lengths = df["text"].apply(lambda x: len(x))
        
        # Get distribution of the scores
        hist = np.histogram(lengths, bins=bins)
        
        if plot:
            plt.stairs(hist[0], hist[1], fill=True)
            plt.xlabel("Review Length")
            plt.ylabel("Frequency")
            plt.show()
        
        return lengths, hist

class VADER(SentimentAnalysis):
    def __init__(self):
        self.classifier = SentimentIntensityAnalyzer()

    def get_sentiment(self, text: (str or pd.DataFrame)):
        if type(text) is str:
            # VADER returns a dict of (pos, neg, neu, compound) scores
            return pd.Series(self.classifier.polarity_scores(text))
        elif type(text) is pd.DataFrame:
            return text['text'].apply(self.get_sentiment)
        else:
            raise Exception("Invalid type for text. Must be str or pd.DataFrame")

    def get_sentiment_distribution(self, df: pd.DataFrame, bins=100, plot=False, score="compound"):
        """
        Overrides the base class method to "compound" score by default. Other options for VADER are "pos", "neg", and "neu".
        """
        scores = self.get_sentiment(df)[score] # compound score takes into account other socres (pos, neg, neu)
        
        # Get distribution of the scores
        hist = np.histogram(scores, bins=bins) # returns (frequencies, bins)
        
        if plot:
            plt.stairs(hist[0], hist[1], fill=True)
            plt.xlabel("Sentiment Score (%s)" % score)
            plt.ylabel("Frequency")
            plt.show()
        
        return scores, hist
        

class TBlob(SentimentAnalysis):
    def __init__(self):
        self.classifier = TextBlob
    
    def get_sentiment(self, text: (str or pd.DataFrame)):
        if type(text) is str:
            # TextBlob returns a tuple of (polarity, subjectivity) scores
            return pd.Series(self.classifier(text).sentiment, ["polarity", "subjectivity"])
        elif type(text) is pd.DataFrame:
            return text['text'].apply(self.get_sentiment)
        else:
            raise Exception("Invalid type for text. Must be str or pd.DataFrame")
        
    
    def get_sentiment_distribution(self, df: pd.DataFrame, bins=100, plot=False, score="polarity"):
        """
        Overrides the base class method to use the polarity score by default. Other option for TextBlob is "subjectivity".
        """
        assert score in ["polarity", "subjectivity"], "Invalid score. Must be 'polarity' or 'subjectivity'"
        
        scores = self.get_sentiment(df)[score] 
        
        # Get distribution of the scores
        hist = np.histogram(scores, bins=bins) # returns (frequencies, bins)
        
        if plot:
            plt.stairs(hist[0], hist[1], fill=True)
            plt.xlabel("Sentiment Score (%s)" % score)
            plt.ylabel("Frequency")
            plt.show()
        
        return scores, hist


class Happy(SentimentAnalysis):
    def __init__(self, model_type="ROBERTA") -> None:
        """
        Uses HappyTransformer to init text classification models from HuggingFace

        Args:
            model_type (str, optional): The type of model to use (ROBERTA, DISTILBERT). Defaults to "ROBERTA".
        """
        # using the top downloaded models: https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
        self.MODELS = {
            "ROBERTA": ["cardiffnlp/twitter-roberta-base-sentiment", 3, 
                        ["LABEL_0", "LABEL_1","LABEL_2"]], # Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
            "DISTILBERT": ["distilbert-base-uncased-finetuned-sst-2-english", 2,
                            ["NEGATIVE", "POSITIVE"]] # Labels
        }
        assert model_type in self.MODELS.keys(), "Invalid model type. Must be one of %s" % self.MODELS.keys()
        self.model_type = model_type
        self.model_info = self.MODELS[model_type]
        self.labels = self.model_info[2]
        self.INPUT_MAX_LEN = 512
        self.classifier = HappyTextClassification(model_type, 
                                             model_name=self.model_info[0],
                                             num_labels=self.model_info[1])
        
    def get_sentiment(self, text: (str or pd.DataFrame)):
        def get_score(result : TextClassificationResult):
            score = 0 # neutral is 0
            assert type(result.score) is float, "Invalid score type. Must be float not %s" % type(result.score)
            
            if result.label == self.labels[0]: # Neg
                score = -result.score
            elif result.label == self.labels[-1]: # Pos
                score = result.score
            return score
            
        if type(text) is str:
            try:
                result = self.classifier.classify_text(text[:self.INPUT_MAX_LEN])
                score = get_score(result)
            except RuntimeError as e:
                score = None            
            return pd.Series(score, ["score"], dtype=float)
        elif type(text) is pd.DataFrame:
            return text['text'].apply(self.get_sentiment)
        else:
            raise Exception("Invalid type for text. Must be str or pd.DataFrame")
        
    def get_sentiment_distribution(self, df: pd.DataFrame, bins=100, plot=False, score='score'):   
        assert score == "score", "Invalid score. Must be 'score'" # only score is available (see get_sentiment)
        
        scores = self.get_sentiment(df)[score] 
        
        # Get distribution of the scores
        hist = np.histogram(scores.dropna(), bins=bins) # returns (frequencies, bins)
        
        if plot:
            plt.stairs(hist[0], hist[1], fill=True)
            plt.xlabel("Sentiment Score (%s)" % score)
            plt.ylabel("Frequency")
            plt.show()
        
        return scores, hist
    
    
