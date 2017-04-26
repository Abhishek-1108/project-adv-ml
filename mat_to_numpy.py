'''
Takes the .mat file as input, Make output file names better
'''
# coding: utf-8

# In[14]:

'''
Converts a .mat file into .npy files 
Give the .mat file as an argument, will save the .npy file in the same 
directory as .mat file

'''
import scipy.io as sio
import cPickle as pickle
import sys

import os.path
def mat_to_numpy(filename):
    mat_contents = sio.loadmat(filename)

    useful_keys = []
    for key in mat_contents:
        if '__' in key:
            continue
        useful_keys.append(key)
        print key

    for key in useful_keys:
        value = mat_contents[key]
        fileToSave = filename.split("/")

        fileExtension = fileToSave[-1].split('.')[0]+'_'+key+'_.npy'
        folder = '/'.join(fileToSave[:-1])
        fname = folder+"/"+fileExtension
        if os.path.isfile(fname):
            fname = raw_input("Please enter name for the file")
            fname = folder +"/"+fname
        with open(fname, "wb") as f:
            pickle.dump(value,f)
            
        print fname+"\t saved"
       

if __name__ == '__main__':
    mat_to_numpy(sys.argv[1])
