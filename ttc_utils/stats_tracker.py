"""
Used to keep track of statistics during training.
Adapted from github.com/caogang/wgan-gp
"""

import os
import numpy as np
import collections
import pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [1]

def load(folder):
    with open(os.path.join(folder, 'training_stats.pkl'), 'rb') as file:
        old_stats = pickle.load(file)
        for stat in old_stats:
            _since_beginning[stat] = old_stats[stat]
    last_tick_ind = max([int(list(_since_beginning[key].keys())[-1]) for key in _since_beginning.keys() if type(_since_beginning[key])==dict])
    offset(last_tick_ind)
    
def tick(): # index incrementer
    _iter[0] += 1

def offset(val): # used when loading from checkpoint
    _iter[0] += val 

def plot(name, value): # record data
    _since_last_flush[name][str(_iter[0])] = value

def flush(folder, return_averages=False): # saves recent data
    stats_averages = {}
  
    for name, vals in _since_last_flush.items():
        stats_averages[name] = np.mean(list(vals.values()))
        _since_beginning[name].update(vals)
    _since_last_flush.clear()

    with open(folder + '/training_stats.pkl', 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
    
    if return_averages:
        return stats_averages