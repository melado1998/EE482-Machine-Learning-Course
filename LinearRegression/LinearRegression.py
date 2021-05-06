"""
the final training loss:  5748300836.287946

"""

import csv
import numpy as np


"""
Read from file and load data in a list

input: File input
output: return a list which contain all data in the file

Author: Waleed Khalid Bin Salman.
version 1.00 11/10/2020
"""
def loadDataset(File_input):
    with open(File_input) as csvfile:
        lines = csv.reader(csvfile)
        next(lines)
        
        return list(lines)


"""
predict a label given values of theta and features

input: theta(can be any number), trainX (our features)
output: return a list which contain all predict labels

Author: Waleed Khalid Bin Salman.
version 1.00 11/10/2020
"""    
def y_predict(theta, trainX):
    return np.dot(theta,trainX)

"""
calculate the error between the true label and predicted label using MSE method

input: true labels, predicted labels 
output: return the total value of error

Author: Waleed Khalid Bin Salman.
version 1.00 11/10/2020
"""
def MSE(y_true, y_predict):
    return np.sum(np.power((y_true-y_predict),2))/len(y_true)


"""
normlize our features

input: features
output: normlized features

Author: Waleed Khalid Bin Salman.
version 1.00 11/10/2020
"""
def normlize(feature, max_value):

    
    return np.array(feature)/max_value


"""
traing our data to draw our best line.

input: training data, learning_rate, limit_changing(between last two losses)

Author: Waleed Khalid Bin Salman.
version 1.00 11/10/2020
"""
def main(test_file,learning_rate,limit_changing):
    
    data= loadDataset(test_file)
    dataload= np.array(data)
    
    feature1= np.array(dataload[:,4],dtype=float)
    max_value_feature1= max(feature1)
    normlize_feature1= normlize(feature1,max_value_feature1)
    
    feature2= np.array(dataload[:,51],dtype=float)
    max_value_feature2= max(feature2)
    normlize_feature2= normlize(feature2,max_value_feature2)
    
    ones= np.ones((len(feature1)),dtype=float)

    train_features= np.array([ones,normlize_feature1,normlize_feature2],dtype=float)
    train_labels= np.array(dataload[:,-1],dtype=float)

    theta= np.zeros((3),dtype=float)
    
    loss=[]
    lr= learning_rate
    lm= limit_changing
    diff_loss= float("inf")
    
    while (diff_loss > limit_changing):
        y_pre=y_predict(theta, train_features) 
        print(y_pre)
        Grad= train_features.dot((y_pre-train_labels).transpose()) / len(train_labels)
        loss.append(MSE(train_labels,y_pre))
        theta= theta - lr*Grad.transpose()
        
        if(len(loss) >= 2):
            diff_loss = abs(loss[-1] - loss[-2])
            
    print("the final training loss: ", str(loss[-1]))
    print(theta)
