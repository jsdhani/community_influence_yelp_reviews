YELP_DATA = "data/raw/yelp/"
YELP_DATA_PATH_FN = lambda x: f"{YELP_DATA}yelp_dataset/yelp_academic_dataset_{x}.json"

YELP_REVIEWS_PATH   = YELP_DATA_PATH_FN("review")
YELP_BUSINESS_PATH  = YELP_DATA_PATH_FN("business")
YELP_USER_PATH      = YELP_DATA_PATH_FN("user")
YELP_TIP_PATH       = YELP_DATA_PATH_FN("tip") # Tips are shorter than reviews and tend to convey quick suggestions.

RESULTS = "results/"
MT_RESULTS_PATH = f"{RESULTS}monte_carlo/"

NETWORK_DATA_PATH = "/data/pandemic/viz/"