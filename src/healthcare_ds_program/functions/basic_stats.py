#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:47:24 2025

@author: hannahgermaine
"""

import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from scipy.stats import linregress

current_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_path)


class run_basic_stats():
    
    def __init__(self,args):
        print('Do Something')
        dataframe = args[0]
        self.savedir = args[1]
        self.predictor_cols = []
        self.run_analysis(dataframe)
        self.run_pairwise_predictors(dataframe)
        
    def run_analysis(self,dataframe):
        print('\n--- Running basic statistical analyses. ---')
        self.stat_dir = os.path.join(self.savedir,'basic_stats')
        if not os.path.isdir(self.stat_dir):
            os.mkdir(self.stat_dir)
        for column in dataframe.columns:
            col_dtype = dataframe[column].dtype
            if col_dtype == 'object':
                self.text_stats(dataframe, column)
            elif len(np.intersect1d(['int64','float64'],col_dtype)) > 0:
                self.predictor_cols.append(column) #Use in machine learning relationships
                self.num_stats(dataframe, column)
            print(' ' + column + ' analyzed.')
    
    def text_stats(self,dataframe,column):
        unique_vals = np.unique(dataframe[column])
        num_unique = len(unique_vals)
        #Calculate repeat histogram and plot
        count_dict = dict(Counter(dataframe[column]))
        if column == 'Name': #Store readmit data
            dataframe['Readmissions'] = dataframe.groupby('Name')['Name'].transform('count')
        text_counts = np.array(list(count_dict.values()))
        repeat_hist = np.histogram(text_counts,bins=np.arange(1,max(text_counts)+2)-0.5)
        repeat_x = np.arange(1,max(text_counts)+1)
        repeat_y = repeat_hist[0]
        #Calculate 99th percentile of repeats
        percentile_99 = np.floor(np.percentile(text_counts,99)).astype('int')
        high_repeat_inds = np.where(text_counts > percentile_99)[0]
        high_repeat_hist = np.histogram(text_counts,bins=np.arange(percentile_99,max(text_counts)+2)-0.5)
        high_repeat_x = np.arange(percentile_99,max(text_counts)+1)
        high_repeat_y = high_repeat_hist[0]
        if num_unique <= 10:
            self.predictor_cols.append(column) #Use in machine learning relationships
            f_stats = plt.figure(figsize=(5,5))
            plt.pie(text_counts,labels=list(count_dict.keys()),\
                              autopct='%1.1f%%')
            plt.title(column + ' Pie')
        else:
            f_stats, ax_stats = plt.subplots(nrows = 1, ncols = 2, figsize=(5,5))
            ax_stats[0].plot(repeat_x,repeat_y,label='_')
            ax_stats[0].set_xlabel('Number of Repeats')
            ax_stats[0].set_xlabel('Number of Occurrences')
            ax_stats[0].set_title('Repeat Overview')
            ax_stats[0].axvline(percentile_99,label='99th Percentile Cutoff',\
                                  linestyle='dashed',color='k',alpha=0.5)
            ax_stats[0].legend(loc='upper right')
            ax_stats[1].plot(high_repeat_x,high_repeat_y)
            ax_stats[1].set_xlabel('Number of Repeats')
            ax_stats[1].set_xlabel('Number of Occurrences')
            ax_stats[1].set_title('>99th Percentile Repeat Overview')
        plt.suptitle(column + ' Statistics')
        plt.tight_layout()
        f_stats.savefig(os.path.join(self.stat_dir,column+'_basic_stats.png'))
        f_stats.savefig(os.path.join(self.stat_dir,column+'_basic_stats.svg'))
        plt.close(f_stats)
        
    def num_stats(self,dataframe,column):
        unique_vals = np.unique(dataframe[column])
        num_unique = len(unique_vals)
        min_val = min(unique_vals)
        max_val = max(unique_vals)
        mean_val = np.mean(dataframe[column])
        median_val = np.median(dataframe[column])
        f_stats = plt.figure(figsize=(5,5))
        plt.hist(dataframe[column],color='b',alpha=0.5,label='_',\
                           bins=np.ceil(max_val-min_val).astype('int'))
        plt.axvline(mean_val,linestyle='dashed',alpha=1,color='k',\
                              label='Mean = ' + str(np.round(mean_val,2)))
        plt.axvline(median_val,linestyle='dashed',alpha=1,color='r',\
                              label='Median = ' + str(np.round(median_val,2)))
        plt.legend(loc='upper right')
        plt.xlabel(column)
        plt.ylabel('Occurrences')
        plt.title(column + ' Distribution')
        plt.tight_layout()
        f_stats.savefig(os.path.join(self.stat_dir,column+'_basic_stats.png'))
        f_stats.savefig(os.path.join(self.stat_dir,column+'_basic_stats.svg'))
        plt.close(f_stats)
        
    def run_pairwise_predictors(self,dataframe):
        print('\n--- Running pairwise linear fit. ---')
        
        pairwise_dir = os.path.join(self.savedir,'pairwise_fits')
        if not os.path.isdir(pairwise_dir):
            os.mkdir(pairwise_dir)
        
        predictor_cols = self.predictor_cols
        num_pred = len(predictor_cols)
        predictor_pairs = list(combinations(np.arange(num_pred),2))
        for p_1,p_2 in tqdm.tqdm(predictor_pairs):
            p_1_n = predictor_cols[p_1]
            p_2_n = predictor_cols[p_2]
            pair_name = p_1_n + ' x ' + p_2_n
            p_1_vals = dataframe[p_1_n]
            p_1_dtype = dataframe[p_1_n].dtype
            if p_1_dtype == 'O':
                p1_unique_types = np.unique(p_1_vals)
                p_1_labels = p_1_vals
                p_1_vals = []
                for p1i in list(p_1_labels):
                    p_1_vals.append(int(np.where(p1i == p1_unique_types)[0][0]))
            p_2_vals = dataframe[p_2_n]
            p_2_dtype = dataframe[p_2_n].dtype
            if p_2_dtype == 'O':
                p2_unique_types = np.unique(p_2_vals)
                p_2_labels = p_2_vals
                p_2_vals = []
                for p2i in list(p_2_labels):
                    p_2_vals.append(int(np.where(p2i == p2_unique_types)[0][0]))
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(p_1_vals, p_2_vals)
            #Plot relationship + fit
            f_pair = plt.figure(figsize=(5,5))
            plt.scatter(p_1_vals, p_2_vals, c = 'blue', alpha=0.5, label='data points')
            #Compare slope against range of y-values
            slope_rat = slope/(np.max(p_2_vals) - np.min(p_2_vals))
            if slope_rat > 0.01:
                # Calculate predicted y values
                p2_pred = slope * np.array(np.unique(p_1_vals)) + intercept
                plt.plot(np.unique(p_1_vals),p2_pred,label='linear fit')
            #else:
                #Unlikely that there's any linear relationship
                #Skip plotting the linear fit
            if p_1_dtype == 'O':
                plt.xticks(np.arange(len(p1_unique_types)),p1_unique_types)
            if p_2_dtype == 'O':
                plt.yticks(np.arange(len(p2_unique_types)),p2_unique_types)
            plt.legend(loc='lower right')
            plt.xlabel(p_1_n)
            plt.ylabel(p_2_n)
            plt.title(pair_name)
            plt.tight_layout()
            f_pair.savefig(os.path.join(pairwise_dir,('_').join(pair_name.split(' ')) + '.png'))
            f_pair.savefig(os.path.join(pairwise_dir,('_').join(pair_name.split(' ')) + '.svg'))
            plt.close(f_pair)
            