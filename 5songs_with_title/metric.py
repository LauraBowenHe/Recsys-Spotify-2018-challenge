# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 10:38:20 2018

@author: bwhe
"""


import numpy as np
import pandas as pd

from collections import OrderedDict



def r_precision(df, max_n_predictions=500):
    targets = df['real']
    predictions = df['pred']
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]

    # Calculate metric
    target_set = set(targets)
    target_count = len(target_set)
    
    return float(len(set(predictions[:target_count]).intersection(target_set))) / target_count

def dcg(relevant_elements, retrieved_elements, k, *args, **kwargs):
    """Compute the Discounted Cumulative Gain.
    Rewards elements being retrieved in descending order of relevance.
    \[ DCG = rel_1 + \sum_{i=2}^{|R|} \frac{rel_i}{\log_2(i + 1)} \]
    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation
    Note: The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.
    Returns:
        DCG value
    """
    retrieved_elements = __get_unique(retrieved_elements[:k])
    relevant_elements = __get_unique(relevant_elements)
    if len(retrieved_elements) == 0 or len(relevant_elements) == 0:
        return 0.0
    # Computes an ordered vector of 1.0 and 0.0
    score = [float(el in relevant_elements) for el in retrieved_elements]
    # return score[0] + np.sum(score[1:] / np.log2(
    #     1 + np.arange(2, len(score) + 1)))
    return np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))


def ndcg(df, k=500, *args, **kwargs):
    r"""Compute the Normalized Discounted Cumulative Gain.
    Rewards elements being retrieved in descending order of relevance.
    The metric is determined by calculating the DCG and dividing it by the
    ideal or optimal DCG in the case that all recommended tracks are relevant.
    Note:
    The ideal DCG or IDCG is on our case equal to:
    \[ IDCG = 1+\sum_{i=2}^{min(\left| G \right|, k)}\frac{1}{\log_2(i +1)}\]
    If the size of the set intersection of \( G \) and \( R \), is empty, then
    the IDCG is equal to 0. The NDCG metric is now calculated as:
    \[ NDCG = \frac{DCG}{IDCG + \delta} \]
    with \( \delta \) a (very) small constant.
    The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.
    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation
    Returns:
        NDCG value
    """
 
    # TODO: When https://github.com/scikit-learn/scikit-learn/pull/9951 is
    # merged...
    relevant_elements = df['real']
    retrieved_elements = df['pred']
    idcg = dcg(
        relevant_elements, relevant_elements, min(k, len(relevant_elements)))
    if idcg == 0:
        raise ValueError("relevent_elements is empty, the metric is"
                         "not defined")
    true_dcg = dcg(relevant_elements, retrieved_elements, k)
    return true_dcg / idcg


def __get_unique(original_list):
    """Get only unique values of a list but keep the order of the first
    occurence of each element
    """
    return list(OrderedDict.fromkeys(original_list))


#Metrics = namedtuple('Metrics', ['r_precision', 'ndcg', 'plex_clicks'])


# playlist extender clicks
def playlist_extender_clicks(df, max_n_predictions=500):
    targets = df['real']
    predictions = df['pred']
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]

    # Calculate metric
    i = set(predictions).intersection(set(targets))
    for index, t in enumerate(predictions):
        for track in i:
            if t == track:
                return float(int(index / 10))
    return float(max_n_predictions / 10.0 + 1)



def coverage(df, max_n_predictions=1000):
    targets = df['real']
    predictions = df['pred']
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]
    
    
    return float(len(set(targets)&set(predictions))) / len(targets)