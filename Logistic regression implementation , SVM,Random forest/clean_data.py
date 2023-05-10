# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    df1 = CTG_features # We implemented in a one liner !!! cool
    c_ctg = {column : pd.to_numeric(df1[column], errors='coerce').dropna() for column in df1.columns if column != extra_feature}
    return c_ctg



def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas dataframe of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe c_cdf containing the "clean" features
    """
    if extra_feature in CTG_features.columns:
        df = CTG_features.drop(extra_feature , axis = 1).copy()
    else:
        df = CTG_features.copy()
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].replace(np.nan, np.random.choice(df[column]))
    c_cdf = df.copy()
    return c_cdf


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_samp
    :return: Summary statistics as a dictionary of dictionaries (called d_summary) as explained in the notebook
    """
    df = c_feat
    d_summary = {column : {'min':df[column].min(),'Q1': np.percentile(df[column],25), 'median': df[column].median(),'Q3':np.percentile(df[column],75) ,'max': df[column].max()} for column in df.columns}

    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_samp
    :param d_summary: Output of sum_stat
    :return: Dataframe containing c_feat with outliers removed

    limits_dict : a dictionary of dictionaries : {column_name{upper limit: value , lower limit: value }}
    we defined limits as follow:
    Lower limit = Q1 - 1.5  IQR
    Upper limit = Q3 + 1.5  IQR
    IQR = Q3 - Q1
    a value outside those limits will be called outliar and will be masked (NAN).

    """

    dict1 = d_summary.copy()
    df = c_feat.copy()
    limits_dict = {column: {'upper': dict1[column]['Q3'] + 1.5 * ((dict1[column]['Q3'] - dict1[column]['Q1'])),
                            'lower': dict1[column]['Q1'] - 1.5 * ((dict1[column]['Q3'] - dict1[column]['Q1']))} for column
                   in df.columns}

    for column in df.columns:
        df[column] = (df[column].mask(df[column] > limits_dict[column]['upper'])) # mask replace values where condition is true with Nan
        df[column] = (df[column].mask(df[column] < limits_dict[column]['lower']))
    c_no_outlier = df.copy()

    return c_no_outlier

def phys_prior(c_samp, feature, thresh):
    """

    :param c_samp: Output of nan2num_samp
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    c_samp[feature] = c_samp[feature].where(c_samp[feature] < thresh)
    c_samp[feature] = c_samp[feature].where(c_samp[feature] > 0)
    filt_feature = c_samp[feature].dropna()
    return np.array(filt_feature)


class NSD:
    def __init__(self):
        self.max = np.nan
        self.min = np.nan
        self.mean = np.nan
        self.std = np.nan
        self.fit_called = False

    def fit(self, CTG_features):
        self.max = CTG_features.max()
        self.min = CTG_features.min()
        self.mean = CTG_features.mean()
        self.std = CTG_features.std()
        self.fit_called = True

    def transform(
        self, CTG_features, mode="none", selected_feat=("LB", "ASTV"), flag=False
    ):
        """
        Note: x_lbl should only be either: 'Original values [N.U]', 'Standardized values [N.U.]', 'Normalized values [N.U.]' or 'Mean normalized values [N.U.]'
        :param mode: A string determining the mode according to the notebook
        :param selected_feat: A two elements tuple of strings of the features for comparison
        :param flag: A boolean determining whether or not plot a histogram
        :return: Dataframe of the normalized/standardized features called nsd_res
        """
        ctg_features = CTG_features.copy()
        if self.fit_called:
            if mode == "none":
                nsd_res = ctg_features
                x_lbl = "Original values [N.U]"
            elif mode == "MinMax":
                nsd_res = (CTG_features - self.min) / (self.max - self.min)
                x_lbl = "MinMax transform values [N.U]"
            elif mode == "standard":
                nsd_res = (CTG_features - self.mean) / self.std
                x_lbl = "standard transform values[N.U]"
            elif mode == "mean":
                nsd_res = (CTG_features - self.mean) / (self.max - self.min)
                x_lbl = "Mean transform values[N.U]"

            if flag:
                self.plot_hist(nsd_res, mode, selected_feat, x_lbl)
            return nsd_res
        else:
            raise Exception("Object must be fitted first!")

    def fit_transform(
        self, CTG_features, mode="none", selected_feat=("LB", "ASTV"), flag=False
    ):
        self.fit(CTG_features)
        return self.transform(
            CTG_features, mode=mode, selected_feat=selected_feat, flag=flag
        )

    def plot_hist(self, nsd_res, mode, selected_feat, x_lbl):
        x, y = selected_feat
        if mode == "none":
            bins = 50
        else:
            bins = 80
            # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        plt.hist(nsd_res[x], bins, alpha=0.5, label=x)
        plt.hist(nsd_res[y], bins, alpha=0.5, label=y)
        plt.xlabel(x_lbl)
        plt.ylabel("Count")
        plt.legend(loc="upper right")
        plt.title(mode)
        plt.show()
        # -------------------------------------------------------------------------


# Debugging block!
if __name__ == '__main__':
    from pathlib import Path
    file = Path.cwd().joinpath(
        'messed_CTG.xls')  # concatenates messed_CTG.xls to the current folder that should be the extracted zip folder
    CTG_dataset = pd.read_excel(file, sheet_name='Raw Data')
    CTG_features = CTG_dataset[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DR', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                                'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance',
                                'Tendency']]
    CTG_morph = CTG_dataset[['CLASS']]
    fetal_state = CTG_dataset[['NSP']]

    extra_feature = 'DR'
    c_ctg = rm_ext_and_nan(CTG_features, extra_feature)