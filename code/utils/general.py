'''
 General utilities
'''
'''
    1) save_pickle                         Function to save files as .npy during workflow
    2) load_pickle                         Function to read files saved during workflows
    
'''

import os
import pickle

#
#------------------------------------------------------------------------
# 1) save pickle file

def pickle_save(name, path, data, verbose=True):
    if not os.path.exists(path):
        os.makedirs(path)
    full_name= (os.path.join(path,name+ '.npy'))


    with open(full_name, 'wb') as f2:
        pickle.dump(data, f2)
    if verbose:
        print('save at: ',full_name)
        
#
#------------------------------------------------------------------------
# 1) load pickle file  

def pickle_load(name, path, verbose=True):  
    #if not os.path.exists(path):
    #    os.makedirs(path)
    full_name= (os.path.join(path,name+ '.npy'))

    with open(full_name, 'r') as f:
        data=pickle.load(f)

    if verbose:
        print('load from: ',full_name)
    return data

def pickle_load2(path2file):
    data = np.load(path2file,allow_pickle=True)
    return data
