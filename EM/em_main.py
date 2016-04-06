'''
Created on Mar 29, 2016

@author: Dhananjay Suresh
'''
import math
import numpy as np
from scipy import stats

def read_file():
    with open("hw2dataset_100.txt", 'rb') as f:
        data_set = np.genfromtxt(f,skip_header=1, missing_values="-", dtype="float")
        weights = np.zeros((data_set.shape[0], data_set.shape[1]+1))
        weights[:,:-1] = data_set
        return weights
        
data = read_file()
'''
0 = Gender
1 = Weight
2 = Height
'''
theta = {0 : [.7, .3],
         1 : [[.8, .2], [.4, .6]],
         2 : [[.7, .3], [.3, .7]]}
theta_ = {0 : [.4, .6],
         1 : [[.5, .5], [.3, .7]],
         2 : [[.3, .7], [.9, .1]]}
THRES = .001
MAX_ITER = 10000
N = data.shape[0]
missing_data_count = np.count_nonzero(np.isnan(data))
missing_data = np.zeros(((missing_data_count*2), data.shape[1]))
M_N = 0
for r in range(N):
    row = data[r]
    if(math.isnan(row[0])):
        y = int(row[1])
        z = int(row[2])
        x0_n = (theta[1][0][y]*theta[2][0][z]*theta[0][0])
        x0_d = theta[1][1][y]*theta[2][1][z]*theta[0][1]+theta[1][0][y]*theta[2][0][z]*theta[0][0]
        x0 = x0_n/x0_d
        x1_n = (theta[1][1][y]*theta[2][1][z]*theta[0][1])
        x1_d = theta[1][0][y]*theta[2][0][z]*theta[0][0]+theta[1][1][y]*theta[2][1][z]*theta[0][1]
        x1 = x1_n/x1_d
        
        missing_data[M_N] = [0, y, z, x0]
        missing_data[M_N+1] = [1, y, z, x1]
        M_N = M_N + 2
        #print("{0}\t{1}".format(x0, x1))
    else:
        data[r][3] = 1
data = data[~np.isnan(data).any(axis=1)]

old_likelihood = 0.0
M_N = missing_data.shape[0]

for i in range(MAX_ITER):
    print "Iteration: {0}".format(i+1)
    r = 0
    while r < M_N:
        row = missing_data[r]
        y = int(row[1])
        z = int(row[2])
        x0_n = (theta[1][0][y]*theta[2][0][z]*theta[0][0])
        x0_d = theta[1][1][y]*theta[2][1][z]*theta[0][1]+theta[1][0][y]*theta[2][0][z]*theta[0][0]
        x0 = x0_n/x0_d
        x1_n = (theta[1][1][y]*theta[2][1][z]*theta[0][1])
        x1_d = theta[1][0][y]*theta[2][0][z]*theta[0][0]+theta[1][1][y]*theta[2][1][z]*theta[0][1]
        x1 = x1_n/x1_d
        #print("{0}\t{1}".format(x0, x1))
        
        missing_data[r] = [0, y, z, x0]
        missing_data[r+1] = [1, y, z, x1]
        r = r + 2

    N = (data[:, 3]).sum(axis=0) + (missing_data[:, 3]).sum(axis=0)
    male = data[data[:,0] == 0]
    weight_male = male[male[:,1] == 0]
    height_male = male[male[:,2] == 0]
    female = data[data[:,0] == 1]
    weight_female = female[female[:,1] == 0]
    height_female = female[female[:,2] == 0]

    male_missing = missing_data[missing_data[:,0] == 0]
    weight_male_missing = male_missing[male_missing[:,1] == 0]
    height_male_missing = male_missing[male_missing[:,2] == 0]
    female_missing = missing_data[missing_data[:,0] == 1]
    weight_female_missing = female_missing[female_missing[:,1] == 0]
    height_female_missing = female_missing[female_missing[:,2] == 0]

    male_N = (male[:,3].sum(axis=0)) + male_missing[:, 3].sum(axis=0)
    female_N = (female[:,3].sum(axis=0)) + female_missing[:, 3].sum(axis=0)

    P_MALE =  male_N/N
    P_FEMALE = 1 - P_MALE
    P_WEIGHT_MALE = ((weight_male[:,3].sum(axis=0)) + (weight_male_missing[:,3].sum(axis=0)))/male_N
    P_WEIGHT_FEMALE = ((weight_female[:,3].sum(axis=0)) + (weight_female_missing[:,3].sum(axis=0)))/female_N
    P_HEIGHT_MALE = ((height_male[:,3].sum(axis=0)) + (height_male_missing[:,3].sum(axis=0)))/male_N
    P_HEIGHT_FEMALE = ((height_female[:,3].sum(axis=0)) + (height_female_missing[:,3].sum(axis=0)))/female_N

    theta_1 = {0 : [P_MALE, P_FEMALE],
               1 : [[P_WEIGHT_MALE, 1 - P_WEIGHT_MALE], [P_WEIGHT_FEMALE, 1 - P_WEIGHT_FEMALE]],
               2 : [[P_HEIGHT_MALE, 1 - P_HEIGHT_MALE], [P_HEIGHT_FEMALE, 1 - P_HEIGHT_FEMALE]]}

    print "P(Male):{0}\t\t\tP(Female):{1}".format(P_MALE, P_FEMALE)
    print "P(Weight>130|Male):{0}\tP(Weight>130|Female):{1}".format(P_WEIGHT_MALE, P_WEIGHT_FEMALE)
    print "P(Height>55|Male):{0}\tP(Height>55|Female):{1}".format(P_HEIGHT_MALE, P_HEIGHT_FEMALE)
    #P(G | H,W)*P(H)*P(W)
    #(G*H*W)/H*W
    H_W = (P_WEIGHT_MALE+P_WEIGHT_FEMALE)*(P_HEIGHT_MALE+P_HEIGHT_FEMALE)
    new_likelihood = (P_WEIGHT_MALE*P_HEIGHT_MALE)*H_W
    if abs(new_likelihood - old_likelihood) < THRES:
        print "Convergence: {0}".format(new_likelihood)
        break;
    print "Likelihood: {0}".format(new_likelihood)
    old_likelihood = new_likelihood
    theta = theta_1
    print ""
    


