'''
Created on Mar 29, 2016

@author: Dhananjay Suresh
'''
import math
import numpy as np

#function to open file
#read data set into numpy array with column for weights
def read_file(file_name):
    with open(file_name, 'rb') as f:
        data_set = np.genfromtxt(f,skip_header=1, missing_values="-", dtype="float")
        weights = np.zeros((data_set.shape[0], data_set.shape[1]+1))
        weights[:,:-1] = data_set
        return weights
#array of dataset filenames
files = ["hw2dataset_10.txt",
         "hw2dataset_30.txt",
         "hw2dataset_50.txt",
         "hw2dataset_70.txt",
         "hw2dataset_100.txt"]
#open csv file for graph data
csv_file = open("likelihood.csv", "w+")  
for f in range(len(files)):
    #read each dataset
    data = read_file(files[f])
    csv_file.write(files[f] + "\n")
    '''
    Original starting parameters
    0 : [Male, Female]
    1 : [[Weight>130|Male, Weight<130|Male], [Weight>130|Female, Weight<130|Female]]
    2 : [[Height>130|Male, Height<130|Male], [Height>130|Female, Height<130|Female]]
    theta is original
    theta_ is changed
    '''
    theta = {0 : [.7, .3],
             1 : [[.8, .2], [.4, .6]],
             2 : [[.7, .3], [.3, .7]]}
    theta_ = {0 : [.8, .2],
             1 : [[.7, .3], [.2, .8]],
             2 : [[.9, .1], [.3, .7]]}
    #threshold
    THRES = .001
    #max number of iterations of EM
    MAX_ITER = 10000
    #size of dataset
    #initialize missing_data with number of rows that have
    #missing values times 2
    N = data.shape[0]
    missing_data_count = np.count_nonzero(np.isnan(data))
    missing_data = np.zeros(((missing_data_count*2), data.shape[1]))
    #index for missing_data array
    M_N = 0
    #set weights for data and missing_data
    for r in range(N):
        row = data[r]
        #if data is missing
        if(math.isnan(row[0])):
            y = int(row[1])
            z = int(row[2])
            #calculate P(G|W,H,theta)
            x0_n = (theta[1][0][y]*theta[2][0][z]*theta[0][0])
            x0_d = theta[1][1][y]*theta[2][1][z]*theta[0][1]+theta[1][0][y]*theta[2][0][z]*theta[0][0]
            x0 = x0_n/x0_d
            #calculate P(G'|W,H,theta)
            x1_n = (theta[1][1][y]*theta[2][1][z]*theta[0][1])
            x1_d = theta[1][0][y]*theta[2][0][z]*theta[0][0]+theta[1][1][y]*theta[2][1][z]*theta[0][1]
            x1 = x1_n/x1_d
            
            #set weight for male and female
            missing_data[M_N] = [0, y, z, x0]
            missing_data[M_N+1] = [1, y, z, x1]
            M_N = M_N + 2
            #print("{0}\t{1}".format(x0, x1))
        else:
            #if data is not missing set weight to 1
            data[r][3] = 1
    #remove all misisng rows from data set
    data = data[~np.isnan(data).any(axis=1)]
    
    old_likelihood = 0.0
    new_likelihood = 0.0
    N = data.shape[0]
    M_N = missing_data.shape[0]
    
    print "STARTING EM ALGORITHM FOR " + files[f]
    
    for i in range(MAX_ITER):
        print "Iteration: {0}".format(i+1)
        csv_file.write(str(i+1)+",")

        r = 0
        
        #E-STEP
        #INFERENCE
        #recalculate weight of missing data using theta
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
            
            #set new weights for male and female values
            missing_data[r] = [0, y, z, x0]
            missing_data[r+1] = [1, y, z, x1]
            r = r + 2
        
        #get total number of data points by adding weights
        total = (data[:, 3]).sum(axis=0) + (missing_data[:, 3]).sum(axis=0)
        #get rows for male, male and weight>130,
        #male and height>130, female, female and height>130
        #for data and missing_data
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
        
        #total number of males and females by summing weights
        male_N = (male[:,3].sum(axis=0)) + male_missing[:, 3].sum(axis=0)
        female_N = (female[:,3].sum(axis=0)) + female_missing[:, 3].sum(axis=0)
        
        #M-STEP
        #Reparameterize
        #expected counts by summing weights over total counts
        P_MALE =  male_N/total
        P_FEMALE = 1 - P_MALE
        P_WEIGHT_MALE = ((weight_male[:,3].sum(axis=0)) + (weight_male_missing[:,3].sum(axis=0)))/male_N
        P_WEIGHT_FEMALE = ((weight_female[:,3].sum(axis=0)) + (weight_female_missing[:,3].sum(axis=0)))/female_N
        P_HEIGHT_MALE = ((height_male[:,3].sum(axis=0)) + (height_male_missing[:,3].sum(axis=0)))/male_N
        P_HEIGHT_FEMALE = ((height_female[:,3].sum(axis=0)) + (height_female_missing[:,3].sum(axis=0)))/female_N
        
        #new parameters
        theta_1 = {0 : [P_MALE, P_FEMALE],
                   1 : [[P_WEIGHT_MALE, 1 - P_WEIGHT_MALE], [P_WEIGHT_FEMALE, 1 - P_WEIGHT_FEMALE]],
                   2 : [[P_HEIGHT_MALE, 1 - P_HEIGHT_MALE], [P_HEIGHT_FEMALE, 1 - P_HEIGHT_FEMALE]]}
    
        print "P(Male):{0}\t\t\tP(Female):{1}".format(P_MALE, P_FEMALE)
        print "P(Weight>130|Male):{0}\tP(Weight>130|Female):{1}".format(P_WEIGHT_MALE, P_WEIGHT_FEMALE)
        print "P(Height>55|Male):{0}\tP(Height>55|Female):{1}".format(P_HEIGHT_MALE, P_HEIGHT_FEMALE)
        
        #calculate likelihood using
        #Sigma P(G | H,W)*P(H)*P(W)
        for r in range(N):
            row = data[r]
            g = int(row[0])
            w = int(row[1])
            h = int(row[2])
            new_likelihood += (theta_1[1][g][w]*theta_1[2][g][h]*theta_1[0][g])
            #new_likelihood += (math.log(theta_1[1][g][w])*math.log(theta_1[2][g][h])*math.log(theta_1[0][g]))
            
        for r in range(M_N):
            row = missing_data[r]
            w = int(row[1])
            h = int(row[2])
            new_likelihood += (theta_1[1][0][w]*theta_1[2][0][h]*theta_1[0][0])+(theta_1[1][1][w]*theta_1[2][1][h]*theta_1[0][1])
            #new_likelihood += (math.log(theta_1[1][0][w])*math.log(theta_1[2][0][h])*math.log(theta_1[0][0]))+(math.log(theta_1[1][1][w])*math.log(theta_1[2][1][h])*math.log(theta_1[0][1]))
            
        #write likelihood data point    
        csv_file.write(str(new_likelihood)+"\n")
        
        #check if any change in new theta and break if there is
        if abs(new_likelihood - old_likelihood) < THRES:
            print "!!!Convergence!!!\nLikelihood: {0}\n".format(new_likelihood)
            break;
        print "Likelihood: {0}".format(new_likelihood)
        #save new likelihood value and new parameters
        old_likelihood = new_likelihood
        new_likelihood = 0.0
        theta = theta_1
        print "\n"
    


