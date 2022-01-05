import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from biosppy.signals import ecg
import scipy
from sklearn.model_selection import KFold
#from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.svm import SVC


# parameters
sampling_rate = 300
folds = 10

# decision tree classifiere
# random_state = 0
# criterion = 'gini'
# splitter = 'best'
# max_depth = 2
# min_samples_split = 2 
# min_samples_leaf = 1
# max_features = 'auto'
# max_leaf_nodes = 2
# min_impurity_decrease = 0

#svc
C = 1
kernel = 'rbf'
gamma = 'auto'
random_state = 0

# functions

# read in data
def read_csv(file_name):
    return pd.read_csv(file_name).values[:,1:]
    

# Find missing values
def missing_values(X):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X)
    return imp.transform(X)

# do sth with nan
def generate_list(matrix):
    sequences = list()
    for row in matrix:
        row = row[~np.isnan(row)]
        sequences.append(row)
    return sequences

def generate_features(X):
    c1 = -1
    nan = []
    H = []
    for row in X:
        c1 += 1
        out = ecg.ecg(signal=row,sampling_rate=sampling_rate, show=False)
        
        # heartbeat features 
        heart_rates = out['heart_rate'][~np.isnan(out['heart_rate'])] #heart rate
        hr_mean = np.mean(heart_rates)#heart rate mean
        if np.isnan(hr_mean):
            nan.append(c1)
            out_new = np.ones(out['heart_rate'].shape[0]+1)*90 # set a random heartbeat of 90, should not occur that often
            hr_mean =  np.mean(out_new) # heart rate mean
            hr_median = np.median(out_new) #heart rate median
            hr_std = np.std(out_new) #heart rate standart deviation
            hr_var = np.var(out_new) #heart rate variance
            hr_mad = scipy.stats.median_absolute_deviation(out_new) #heart rate median absloute deviation
        else:
            hr_median = np.median(heart_rates) #heart rate median
            hr_std = np.std(heart_rates) #heart rate standart deviation
            hr_var = np.var(heart_rates) #heart beat variance
            hr_mad = scipy.stats.median_absolute_deviation(heart_rates) #heart rate median absloute deviation
        hr_features = np.array([hr_mean,hr_median,hr_std,hr_var, hr_mad]) #first few features

        r_intensities = row[out['rpeaks']]
        r_int_mean = np.mean(r_intensities) #mean of the intrnsities
        r_int_median = np.median(r_intensities) # median of the intensities
        r_int_std = np.std(r_intensities) #stadnard deviation  of the intensities
        r_int_var = np.var(r_intensities) #variance of the intensities
        r_int_mad = scipy.stats.median_absolute_deviation(r_intensities) #median absolute deviation of the intensities
        r_int_features = np.array([r_int_mean,r_int_median,r_int_std,r_int_var, r_int_mad]) #more features
        
        #RR interval features 
        r_peaks = out['ts'][out['rpeaks']] 
        rri = np.diff(r_peaks)*1000 # rr interval in milli seconds
        rri_mean = np.mean(rri) #mean of the rr intervals
        rri_median = np.median(rri) #median of the rr intervals
        rri_std = np.std(rri) #standard deviation of the rr intervals
        rri_var = np.var(rri) # variance of the rr intervals
        rri_diff = np.absolute(np.diff(rri)) #difference of peaks: out[i] = a[i+1] - a[i]
        nn10 = rri_diff[rri_diff>10].shape[0] #amount of peak diff higher than 10
        nn20 = rri_diff[rri_diff>20].shape[0] #amount of peak diff higher than 20
        nn50 = rri_diff[rri_diff>50].shape[0] #amount of peak diff higher than 50
        nn100 = rri_diff[rri_diff>100].shape[0] #amount of peak diff higher than 100
        nn200 = rri_diff[rri_diff>200].shape[0] #amount of peak diff higher than 200
        nn500 = rri_diff[rri_diff>500].shape[0] #amount of peak diff higher than 500
        rri_diff_length = rri_diff.shape[0]
        pnn10 = nn10 / rri_diff_length #amount of peak diff higher than 10 compared to the number all differences
        pnn20 = nn20 / rri_diff_length #amount of peak diff higher than 20 compared to the number all differences
        pnn50 = nn50 / rri_diff_length #amount of peak diff higher than 50 compared to the number all differences
        pnn100 = nn100 / rri_diff_length #amount of peak diff higher than 100 compared to the number all differences
        pnn200 = nn200 / rri_diff_length #amount of peak diff higher than 200 compared to the number all differences
        pnn500 = nn500 / rri_diff_length #amount of peak diff higher than 500 compared to the number all differences
        rmssd = np.sqrt(np.mean(rri_diff**2)) #root mean squared
        cvsd = rmssd / rri_mean #coefficient of variance
        sdsd = np.std(rri_diff) #standard deviation
        madnn = scipy.stats.median_absolute_deviation(rri) #median absolute deviation
        mcvnn = madnn / rri_median #median coefficient variance
           
        r_features = np.array([r_int_mean,r_int_median,r_int_std,r_int_var, r_int_mad, rri_mean, rri_median, rri_std, rri_var, nn10, nn20, nn50, nn100, nn200, nn500, pnn10, pnn20, pnn50, pnn100, pnn200, pnn500, rmssd, cvsd, sdsd, madnn, mcvnn]) #even more features
        
        if np.any(np.isnan(r_features)): #should never be the case
            nan.append(c1)
            r_features = np.zeros(r_features.shape[0])
        
        features = np.concatenate((hr_features, r_int_features,r_features), axis=None)
        H.append(features)
        G = np.asarray(H)
    return G
        
        
def ml_predict(X_train,X_test,y):
    kf = KFold(n_splits=folds, shuffle = True, random_state = random_state)
    prediction = np.zeros(X_test.shape[0])
    for train_index, test_index in kf.split(X_train):
        X_training, X_testing = X_train[train_index], X_train[test_index]
        y_train, y_test = y[train_index].ravel(), y[test_index].ravel()  
        
        #clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease).fit(X_training, y_train)
        clf = SVC(C=C, kernel=kernel, gamma=gamma, random_state=random_state).fit(X_training, y_train)
        y_pred = clf.predict(X_testing)
        prediction += clf.predict(X_test)
        
        score = f1_score(y_test, y_pred, average='micro')
        print(score)
    y_pred = prediction/folds
    y_pred = np.round_(y_pred,decimals=0, out=None)
    return y_pred

# main
X_train = read_csv('X_train.csv')
X_test = read_csv('X_test.csv')
y = read_csv('y_train.csv')

X_train = generate_list(X_train)
X_test = generate_list(X_test)

X_train = generate_features(X_train)
X_test = generate_features(X_test)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)   
X_test = scaler.transform(X_test)
y_pred = ml_predict(X_train, X_test, y)

# output
filename = 'submatt.csv'
with open(filename, 'w') as output_file:
    output_file.write('id,y\n')
    for i in range(3411):
        output_file.write(str(i))
        output_file.write(',')
        output_file.write(str(y_pred[i]))
        output_file.write('\n')