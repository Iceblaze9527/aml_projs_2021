import numpy as np
from sklearn.linear_model import Ridge
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math


with open('train.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))


#write solution
csvfile=open('submission.csv','w', newline='')
obj=csv.writer(csvfile)
for i in range (0,5):
    obj.writerow([RMSE[i]])
csvfile.close()
