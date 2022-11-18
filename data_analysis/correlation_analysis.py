"""
This file is implementing the correlation analysis. from the 2017 paper including:
    * Pearson's correlation coefficient
    * P-value
    * Regression analysis via linear regression     
"""
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats._result_classes import PearsonRResult
import pandas as pd
import numpy as np


def get_pearson(data_dict:dict, alt:str='less'):
    """
    This function is calculating the Pearson's correlation coefficient which 
    "measures the linear relationship between two datasets".
    
    args:
        data_dict: dictionary of {x:y} values for the correlation analysis.
        alt: alternative hypothesis. Default is 'less' which means that the 
            correlation is negative.
        
    return: 
        PearsonRResult: Pearson's correlation result (statistic, pvalue), 
            can also be used to specify convidence intervals.
    """
    x, y = zip(*sorted(data_dict.items()))
    return pearsonr(x, y, alternative=alt)