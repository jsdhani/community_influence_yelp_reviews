YELP_DATA = "data/raw/yelp/"
YELP_DATA_PATH_FN = lambda x: YELP_DATA + "yelp_dataset/yelp_academic_dataset_{}.json".format(x)

YELP_REVIEWS_PATH   = YELP_DATA_PATH_FN("review")
YELP_BUSINESS_PATH  = YELP_DATA_PATH_FN("business")
YELP_USER_PATH      = YELP_DATA_PATH_FN("user")
YELP_TIP_PATH       = YELP_DATA_PATH_FN("tip") # Tips are shorter than reviews and tend to convey quick suggestions.

