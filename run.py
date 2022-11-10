"""
I am just using this as a test bed for now, please excuse the mess...
"""

# %%
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.query_raw_yelp import QueryYelp as qy
from utils.sentiment import VADER, TBlob, Happy
from common.config_paths import YELP_DATA
import pandas as pd

# %% file location:
REVIEW = YELP_DATA + "yelp_dataset/yelp_academic_dataset_review.json"

# %% get the data:
reader = qy.get_json_reader(REVIEW)
# combining multiple chunks into one dataframe
df_rev = next(reader) # 1000 rows
for i in range(500): # 10,000 rows
    chunk = next(reader)
    df_rev = pd.concat([df_rev, chunk], ignore_index=True)
    

#######################################################
# %% Happy Transformer:
hp = Happy(model_type="DISTILBERT")


# %% get the sentiment of 10 reviews:
# pd.set_option("display.max_colwidth", None)
# pd.set_option('display.colheader_justify', 'right')
# res = pd.concat([df_rev['text'][:10], df_rev['stars'][:10], hp.get_sentiment(df_rev[:10])], axis=1, ignore_index=True)
# %% get distribution of sentiment scores:
_ = hp.get_sentiment_distribution(df_rev, bins=100, plot=True)

#######################################################
# %% VADER:
v_mdl = VADER()

length, dist_l = v_mdl.get_review_length_distribution(df_rev, bins=100, plot=True)
scores, dist_s = v_mdl.get_sentiment_distribution(df_rev, bins=100, plot=True)

# %% splitting into short reviews and long reviews
cf = 200
df_rev_short = df_rev[df_rev['text'].str.len() < cf]
df_rev_long = df_rev[df_rev['text'].str.len() >= cf]

print(len(df_rev_short), len(df_rev_long))

# %% displaying sentiment distribution with short reviews
scores_short, dist_short = v_mdl.get_sentiment_distribution(df_rev_short, bins=100, plot=True)

# %% displaying sentiment distribution with long reviews
scores_long, dist_long = v_mdl.get_sentiment_distribution(df_rev_long, bins=100, plot=True)


# %% calculating correlation between star rating and sentiment
stars = df_rev['stars']
data = pd.concat([stars, scores, length], axis=1)
print(data.groupby('stars').mean())
print(data.corr())



#######################################################
# %% TEXTBLOB:
tb_mdl = TBlob()

scores, dist_s = tb_mdl.get_sentiment_distribution(df_rev, bins=100, plot=True)
_ = tb_mdl.get_sentiment_distribution(df_rev, bins=100, plot=True, score="subjectivity")


# %%
