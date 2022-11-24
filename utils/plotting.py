from data_analysis.monte_carlo import ReviewProb
from data_analysis.rating_analysis import RatingAnalysisFriends, RatingAnalysisGeneral
from common.config_paths import (
            YELP_REVIEWS_PATH, YELP_USER_PATH, 
            MC_RESULTS_PATH, RESULTS, RATINGS_CORR_PATH)
from common.constants import restaurant_categories
from utils.query_raw_yelp import QueryYelp as qy
from data_analysis.correlation_analysis import get_pearson, get_linear_reg
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import math

def get_pkl_path(dir_path, time_period):
    t_s = f"{time_period[0].strftime('%Y-%m-%d')}_{time_period[1].strftime('%Y-%m-%d')}"
    path = lambda x: f"{dir_path}{t_s}{x}.pkl"
    return path, t_s

def get_mc_pkl_path(dir_path, time_period):
    # path -> {dir_path}/{t_s}_prob_0.pkl
    path = lambda x: get_pkl_path(dir_path, time_period)(f"_prob_{x}")
    
    data_0 = pickle.load(open(path(0), "rb"))
    data_1 = pickle.load(open(path(1), "rb"))
    
    return data_0, data_1
    
def bin_data(data, bins=[x for x in range(0,51,5)], ignore_exact=[0,1], normalize=False):
    """
    This function is binning the data into bins number of bins.
    """
    binned_d = {k: 0 for k in bins}
    for i in range(len(bins)):
        l = bins[i]
        r = bins[i+1] if i < len(bins)-1 else math.inf # last bin is open ended
        for k,v in data.items():
            if ignore_exact and k in ignore_exact: continue
            if l <= k and k < r:
                binned_d[l] += v
                
    # normalize
    if normalize:
        binned_d = {k: v/sum(binned_d.values()) for k,v in binned_d.items()}
    return binned_d

def plot_bins(binned_d0, binned_d1=None, w=2.5):
    """_summary_

    Args:
        binned_d0 (dict): {bin: count}
        binned_d1 (_type_, optional): _description_. Defaults to None.
    """
    if binned_d1 is None:
        data = np.array([[x,y] for x,y in binned_d0.items()])
        plt.bar(data[:,0], data[:,1], width=5, align='edge')
    else: # grouped bar chart
        data_0 = np.array([[x,y] for x,y in binned_d0.items()])
        data_1 = np.array([[x,y] for x,y in binned_d1.items()])
        
        h0 = plt.bar(data_0[:,0], data_0[:,1], width=-w, 
                        align='edge', tick_label=data_0[:,0])
        h1 = plt.bar(data_1[:,0], data_1[:,1], width=w, 
                        align='edge')
        
        plt.legend((h0[0], h1[0]), ('P(0|i)', 'P(1|i)'))

def plot_mc_prob(time_periods):
    MC_PATH_PKL = "results/monte_carlo_prob0/"
    MC_PATH_MEDIA = "media/monte_carlo_prob01_binned/"
    bins = [x for x in range(0,51,5)]
    ignore_exact = [0,1]
    for t in time_periods[:5]:
        path, t_s = get_pkl_path(MC_PATH_PKL, t)
        
        data_0 = pickle.load(open(path("_prob_0"), "rb"))
        data_1 = pickle.load(open(path("_prob_1"), "rb"))
        
        bd_0 = bin_data(data_0, bins, ignore_exact)
        bd_1 = bin_data(data_1, bins, ignore_exact)
        
        plot_bins(bd_1, bd_0)
        
        plt.xlabel("Number of i friends who reviewed same business")
        plt.ylabel("Monte Carlo probability")
        plt.title(f"{t_s}: Ignoring i={ignore_exact}")
        
        i_str = "".join([str(x) for x in ignore_exact])
        plt.savefig(f"{MC_PATH_MEDIA}{t_s}_prob_01_i{i_str}.png")
        plt.show()
        plt.clf()

def plot_over_time(data, time_periods,ax=None,idx=0, labels=('P(0|i)', 'P(1|i)')): # idx picks the elemnt of the tuple in data
    """_summary_

    Args:
        data (list of pears): must be of shape (t,2,2) where t = len(time_periods)
            * for one time period data is of shape (2,2)
                * rows are for P(0|i) and P(1|i)
        time_periods (_type_): _description_
    """
    w = 0.4
    data = np.array(data)
    y_pos = list(range(len(data)))
    d_0 = data[:, 0, idx]
    d_1 = data[:, 1, idx]
    if ax is None:
        fig, ax = plt.subplots()
    h0 = ax.bar([y-w/2 for y in y_pos], d_0[::-1], width=w, align='center')
    h1 = ax.bar([y+w/2 for y in y_pos], d_1[::-1], width=w, align='center')
    plt.legend((h0[0], h1[0]), labels, loc='upper left')
    _=ax.set_xticks(y_pos,
        labels=[f"{p[0].strftime('%Y')}-{p[1].strftime('%Y')}" for p in time_periods][::-1])

def plot_mc_prob_compare(time_periods, ignore_exact=[0,1], 
                         bins=[x for x in range(0,51,5)], save_path=None):
    MC_PATH_PKL = "results/monte_carlo_prob0/"
    MC_PATH_MEDIA = "media/monte_carlo_prob01_binned/"
    pears = []
    lines = []
    print("{:25}|{:^10}|{:^10}|{:^10}|{:^10}".format(
            "period", "coeff", "p-value", "slope","intercept"))
    print("-"*55)
    for t in time_periods:
        path, t_s = get_pkl_path(MC_PATH_PKL, t)
        data_0 = pickle.load(open(path("_prob_0"), "rb"))
        data_1 = pickle.load(open(path("_prob_1"), "rb"))
        bd_0 = bin_data(data_0, bins, ignore_exact)
        bd_1 = bin_data(data_1, bins, ignore_exact)
        
        # calculating pearson correlation and linear regression
        pear_0 = get_pearson(bd_0, alt="two-sided")
        line_0 = get_linear_reg(bd_0)
        
        pear_1 = get_pearson(bd_1, alt="two-sided")
        line_1 = get_linear_reg(bd_1)
        
        pear = [pear_0, pear_1]
        line = [line_0, line_1]
        
        pears.append(pear)
        lines.append(line)
        
        print("{:25}|{:^10.3}|{:^10.3}|{:^10.3}|{:^10.3}".format(
            t_s,pear[0][0]-pear[1][0], # coeff diff
                pear[0][1]-pear[1][1], # p-value diff
                line[0][0]-line[1][0], # slope diff
                line[0][1]-line[1][1], )) # intercept diff
    
    plot_over_time(pears, time_periods)
    plt.title("Monte Carlo Probability Correlation")
    plt.xlabel("Time Period")
    plt.ylabel("Pearson Correlation Coefficient")
    
    if save_path:
        plt.savefig(save_path+"/all_corr.png")
    plt.clf()
    
    plot_over_time(lines, time_periods)
    plt.title("Monte Carlo Probability Regression")
    plt.xlabel("Time Period")
    plt.ylabel("Line Slope")

    if save_path:
        plt.savefig(save_path+"/all_lin.png")
    plt.clf()
        
    return pears, lines

def plot_rating_dist(values:list, t_s, section="Restaraunt", save_path=None, alpha=0.5):
    bins = np.histogram(values, bins=10, density=True)
    plt.stairs(bins[0], bins[1], fill=True, label=t_s, alpha=alpha)
    plt.yticks(ticks=[])
    plt.xlabel("Star Rating")
    plt.title(f'Average {section} Rating Distribution \n{t_s}')
    
    if save_path:
        plt.savefig(f'{save_path}{t_s}.png')

def plot_ratings(ratings:np.ndarray, t_s, section='All Ratings', line=None, save_path=None): # line is a tuple of (slope, intercept)
    plt.scatter(ratings[:,0], ratings[:,1])
    
    if line is not None:
        plt.plot(ratings[:,0], line[0]*ratings[:,0] + line[1], label=t_s, color='k')
    
    plt.xlabel("User Rating")
    plt.ylabel("Average Friend Rating")
    plt.title(f"{section}\n{t_s}")
    # plt.legend()
    
    if save_path:
        plt.savefig(f'{save_path}{t_s}.png')
    plt.show()

def get_time_periods(covid_range=(pd.Timestamp('2019-12-01'), pd.Timestamp('2021-08-01')), 
                     num_periods=4):
    len_covid = covid_range[1] - covid_range[0] # 609 days
    time_periods = []
    for x in range(num_periods):
        start = covid_range[0] - (x * len_covid)
        end = covid_range[1] - (x * len_covid)
        time_periods.append((start, end))
    return time_periods

if __name__ == "__main__":
    time_periods = get_time_periods()
    
    
    # Plotting rating distributions for businesses
    path=f'{RESULTS}ratings-overall-no-rest/ratings_'
    save_path='media/final_ptt_plots/ratings/distributions/no-rest/'

    for i, t in enumerate(time_periods):
        curr_path, t_s = get_pkl_path(path, t)
        data = pickle.load(open(curr_path(""), 'rb'))

        plot_rating_dist(list(data.values()), t_s, 
                        section='Restaurant', alpha=.5)#, save_path=save_path)
        # plt.show()
        # plt.clf()
    plt.title(f"Rating Distribution for Non-Restaurants")
    plt.legend(loc='upper left')
    plt.savefig(f'{save_path}overlay.png')
        
    # plotting rating corr:
    PKL_FOLDER_PATH = f"{RESULTS}ratings-no-rest/ratings_"
    SAVE_PATH = "media/final_ptt_plots/ratings/friend_corr/no-rest/"


    # %% plotting with linear regression
    # # FOR RATINGS
    for i,t in enumerate(time_periods):
        path, t_s = get_pkl_path(PKL_FOLDER_PATH, t)
        data = pickle.load(open(path(''), 'rb'))
        line = get_linear_reg(data)
        plot_ratings(data, t_s, section="No Restaurants",
                    line=line, save_path=SAVE_PATH)
        
    # ratings lines
    PKL_FOLDER_PATH = f"{RESULTS}ratings/ratings_"
    SAVE_PATH = "media/final_ptt_plots/ratings/friend_corr/all/"


    # %% plotting with linear regression
    # # FOR RATINGS
    lines = []
    for i,t in enumerate(time_periods):
        path, t_s = get_pkl_path(PKL_FOLDER_PATH, t)
        data = pickle.load(open(path(''), 'rb'))
        line = get_linear_reg(data)
        lines.append(line)
        
        plt.plot(data[:,0], data[:,0]*line[0] + line[1], label=f"{t_s}")
        

    plt.xlabel("User Rating")
    plt.ylabel("Average Friend Rating")
    plt.ylim(0.8,5.2)
    plt.xlim(0.8,5.2)
    plt.title("Average Friend Rating Regression Lines (All Businesses)")
    plt.legend()
    # plt.savefig(f"{SAVE_PATH}lines.png")