"""
I am just using this as a test bed for now, please excuse the mess...
"""

# %%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.query_raw_yelp import QueryYelp as qy
from utils.sentiment import VADER_SentimentAnalysis
from common.config_paths import YELP_DATA
import pandas as pd

# %% file location:
REVIEW = YELP_DATA + "yelp_dataset/yelp_academic_dataset_review.json"

# %%
v_mdl = VADER_SentimentAnalysis()

reader = qy.get_json_reader(REVIEW)
df_rev = next(reader) # 1000 rows

# %% displaying sentiment distribution
scores = v_mdl.get_sentiment_distribution(df_rev, plot=True)


# %% merge the scores with text and sort by compound score
# text = df_rev["text"]
# df = pd.concat([text, scores], axis=1)

# df_sort = df.sort_values("compound") # +1 compound value indicates most positive sentiment

# %% get the top 10 most positive and 10 most negative reviews and display
# for i in range(10):
#     print("-" * 50)
#     print(df_sort["compound"].iloc[i])
#     print(df_sort["text"].iloc[i])
#     print("-" * 50)
    
    

# def get_top_n_reviews(self, df: pd.DataFrame, n: int, sort_col: str, ascending: bool = False):
#     return df.sort_values(sort_col, ascending=ascending).iloc[:n]["text"]

# def get_bottom_n_reviews(self, df: pd.DataFrame, n: int, sort_col: str, ascending: bool = False):
#     return df.sort_values(sort_col, ascending=ascending).iloc[-n:]["text"]
# for i in range(10):
#     print("-" * 50)
#     print(df_sort["compound"].iloc[-i])
#     print(df_sort["text"].iloc[-i])
#     print("-" * 50)

# # %%
# max_score = 0
# target = 'cool'
# text = "EMPTY"
# for chunk in reader:
#     row = chunk.sort_values(target)[::-1].iloc[0,4:8]
#     if max_score < row[target]:
#         max_score = row[target]
#         text = row['text']
# print(max_score)
# print(text)

# %%
