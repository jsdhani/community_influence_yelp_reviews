"""
A couple pre-trained sentiment analysis models that I will test:
* Vader
* TextBlob - No paper for this one, but it is a very popular library
* Happy Transformer
"""
import torch
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
        
    def get_sentiment_gpu(self, text: pd.DataFrame, batch_size=128):
        """
        Uses GPU to get sentiment. Requires CUDA enabled GPU.
        """
        
        self.classifier._pipeline._batch_size = batch_size
        result = self.classifier._pipeline( # returns list of dicts with keys [label, score]
            text['text'].str.slice(0,self.INPUT_MAX_LEN).tolist()) 
        
        def get_score(result : dict):
            assert type(result['score']) is float, "Invalid score type. Must be float not %s" % type(result.score)
            score = 0 # neutral is 0
            
            if result['label'] == self.labels[0]: # Neg
                score = -result['score']
            elif result['label'] == self.labels[-1]: # Pos
                score = result['score']
            return score
        
        return pd.Series([get_score(x) for x in result])
    
    def get_sentiment_distribution(self, df: pd.DataFrame, bins=100, plot=False, score='score', use_cuda=True):
        assert score == "score", "Invalid score. Must be 'score'" # only score is available (see get_sentiment)
        
        if use_cuda:
            assert torch.cuda.is_available(), "CUDA is not available. Cannot use GPU"
            scores = self.get_sentiment_gpu(df)
        else:
            scores = self.get_sentiment(df)[score]
        
        # Get distribution of the scores
        hist = np.histogram(scores.dropna(), bins=bins) # returns (frequencies, bins)
        
        if plot:
            plt.stairs(hist[0], hist[1], fill=True)
            plt.xlabel("Sentiment Score (%s)" % score)
            plt.ylabel("Frequency")
            plt.show()
        
        return scores, hist
    
    
    
if __name__ == "__main__":
    from common.config_paths import YELP_REVIEWS_PATH
    from utils.query_raw_yelp import QueryYelp as qy
    import pandas as pd
    # get the data:
    reader = qy.get_json_reader(YELP_REVIEWS_PATH)
    # combining multiple chunks into one dataframe
    df_rev = next(reader) # 1000 rows
    for i in range(500): # 10,000 rows
        chunk = next(reader)
        df_rev = pd.concat([df_rev, chunk], ignore_index=True)
        

    #######################################################
    # Happy Transformer:
    hp = Happy(model_type="DISTILBERT")


    # get the sentiment of 10 reviews:
    # pd.set_option("display.max_colwidth", None)
    # pd.set_option('display.colheader_justify', 'right')
    # res = pd.concat([df_rev['text'][:10], df_rev['stars'][:10], hp.get_sentiment(df_rev[:10])], axis=1, ignore_index=True)
    # get distribution of sentiment scores:
    _ = hp.get_sentiment_distribution(df_rev, bins=100, plot=True)

    #######################################################
    # VADER:
    v_mdl = VADER()

    length, dist_l = v_mdl.get_review_length_distribution(df_rev, bins=100, plot=True)
    scores, dist_s = v_mdl.get_sentiment_distribution(df_rev, bins=100, plot=True)

    # splitting into short reviews and long reviews
    cf = 200
    df_rev_short = df_rev[df_rev['text'].str.len() < cf]
    df_rev_long = df_rev[df_rev['text'].str.len() >= cf]

    print(len(df_rev_short), len(df_rev_long))

    # displaying sentiment distribution with short reviews
    scores_short, dist_short = v_mdl.get_sentiment_distribution(df_rev_short, bins=100, plot=True)

    # displaying sentiment distribution with long reviews
    scores_long, dist_long = v_mdl.get_sentiment_distribution(df_rev_long, bins=100, plot=True)

    # calculating correlation between star rating and sentiment
    stars = df_rev['stars']
    data = pd.concat([stars, scores, length], axis=1)
    print(data.groupby('stars').mean())
    print(data.corr())

    #######################################################
    # TEXTBLOB:
    tb_mdl = TBlob()

    scores, dist_s = tb_mdl.get_sentiment_distribution(df_rev, bins=100, plot=True)
    _ = tb_mdl.get_sentiment_distribution(df_rev, bins=100, plot=True, score="subjectivity")

