import pandas as pd
import numpy as np
import itertools

def drop_duplicates(test_): # drop any numbers that were duplicated in a row.
    new = test_.transpose().reset_index(drop=True)
    col = new.columns.to_list()
    L =[]
    for c in col:
        l = new[c].unique()
        
        if len(l)<6:
            L.append(c)
    t = new.drop(L, axis=1).transpose()
    print("Data clean is done...")
    return t
def range_select(df, std = 3):
    rolling4_max = [12,	24,	37,	47,	70] # check N1-N5, the max value for each field is in this rolling4 mean list. min is previous value
    rolling4_min = [1, 12, 24, 37, 58]
    std = std # allow the max value having difference of 3
    col = df.columns.to_list()
    drop_index = []
    for c in range(len(col)-1): # find the index that is not meeting the criteria, then drop those index later
        ind_max = df.loc[df[col[c]] > rolling4_max[c]+std].index.to_list()
        #print(ind_max)
        ind_min = df.loc[df[col[c]] < rolling4_min[c]-std].index.to_list()
        drop_index.append(ind_max)
        drop_index.append(ind_min)
    flat_list = list(set(list(itertools.chain.from_iterable(drop_index))))
    #print(flat_list)
    selected = df.drop(flat_list,axis=0)
    selected.columns = col
    return flat_list, selected

