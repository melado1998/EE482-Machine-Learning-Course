"""
When K= 1 using L2 distance, The Average Accuracy is=  0.96
When K= 3 using L2 distance, The Average Accuracy is=  0.96
When K= 5 using L2 distance, The Average Accuracy is=  0.9600000000000002
When K= 7 using L2 distance, The Average Accuracy is=  0.9733333333333334
--- 0.2872300148010254 seconds ---

--------------------------------------------------------------------------

When K= 1 using L1 distance, The Average Accuracy is=  0.9466666666666667
When K= 3 using L1 distance, The Average Accuracy is=  0.9666666666666666
When K= 5 using L1 distance, The Average Accuracy is=  0.9600000000000002
When K= 7 using L1 distance, The Average Accuracy is=  0.9533333333333334
--- 0.21747612953186035 seconds ---

------------------------------------------------------------------------------------

Maximum accuracy When K= 7 using L2 distance, The Average Accuracy is=  0.9733333333333334

"""
import csv
import math
import random
import time
import numpy as np



"""
Read from file and load data in a list

input: File input
output: return a list which contain all data in the file

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
"""
def loadDataset(File_input):
    with open(File_input) as csvfile:
        lines = csv.reader(csvfile)
        next(lines)
        
        return list(lines)
    

"""
shaffle the data inside a list

input: list
output: return a shuffled list 

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
"""    

def shuffle_data(data):
    shuffled_data=random.sample(data,len(data))
    
    return shuffled_data

"""
split the data list with a specific percent 

input: list, split percent
output: return a splited list 

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
"""   
def split_data(shuffled_data,persent_of_test):
    split = [shuffled_data[x:x+int((len(shuffled_data)*(persent_of_test/100)))] for x in range(0, len(shuffled_data), int((len(shuffled_data)*(persent_of_test/100))))]
    
    return split

"""
group a split data into a train and test data

input: list
output: return list with groups of train and test data

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
"""   
def Train_Test_data(divided_data):
    Test_data1=divided_data[0]
    Train_data1=divided_data[1]+divided_data[2]+divided_data[3]+divided_data[4]
    ######################################
    Test_data2=divided_data[1]
    Train_data2=divided_data[0]+divided_data[2]+divided_data[3]+divided_data[4]
    ######################################
    Test_data3=divided_data[2]
    Train_data3=divided_data[0]+divided_data[1]+divided_data[3]+divided_data[4]
    ######################################
    Test_data4=divided_data[3]
    Train_data4=divided_data[0]+divided_data[1]+divided_data[2]+divided_data[4]
    ######################################
    Test_data5=divided_data[4]
    Train_data5=divided_data[0]+divided_data[1]+divided_data[2]+divided_data[3]
    ######################################
    Train_Test_data_list=[[Test_data1,Train_data1],[Test_data2,Train_data2],
                          [Test_data3,Train_data3],[Test_data4,Train_data4],
                          [Test_data5,Train_data5]]
    
    return Train_Test_data_list

"""
calculate the Euclidean_distance

input: information of train element, infomation of test element, lenght of informations
output: return the Euclidean_distance

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
"""  
def Euclidean_distance(Train,Test,lenght):
    distance=0
    print("euclidean",Train)
    for x in range(1,lenght):
        distance+=pow(float(Train[x])-float(Test[x]),2)
    
    return math.sqrt(distance)

"""
calculate the Manhattan distance

input: information of train element, infomation of test element, lenght of informations
output: return the Manhattan_distance

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
"""  
def Manhattan_distance(Train,Test,lenght):
    distance=0
    for x in range(1,lenght):
        distance+=abs(float(Train[x])-float(Test[x]))
    
    return distance


"""
calculate the Euclidean distance from test element to all the training elements

input: training data set, test element
output: return a list with the distance from test element to training elements

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
"""  
def Neighbors_dist_L2(training_data, Test_element):
    neighbors_dist=[]
    print('dl2',Test_element)
    lenght=len(Test_element)-1
    for x in training_data:
        dist=Euclidean_distance(x,Test_element,lenght)
        neighbors_dist.append((x,dist))
        
    neighbors_dist.sort(key = lambda x:x[1])
    return neighbors_dist

"""
calculate the Manhattan distance from test element to all the training elements

input: training data set, test element
output: return a list with the distance from test element to training elements

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
"""  
def Neighbors_dist_L1(training_data, Test_element):
    neighbors_dist=[]
    lenght=len(Test_element)-1
    for x in training_data:
        dist=Manhattan_distance(x,Test_element,lenght)
        neighbors_dist.append((x,dist))
        
    neighbors_dist.sort(key = lambda x:x[1])
    return neighbors_dist


"""
extract the K closest neighbors from a list

input: list, number of closest neighbors
output: return a list of closest neighbors

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
""" 
def K_closest_neighbors(sorted_array,K):
    closest_neighbors=[]
    for x in range(K):
        closest_neighbors.append(sorted_array[x])
        
    return closest_neighbors

"""
classify test element

input: list of closest neighbors
output: return the class of the element

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
""" 
def classify_item(closest_neighbors):
    classVotes={}
    for x in closest_neighbors:
        item_class=x[0][-1]
        if item_class in classVotes:
            classVotes[item_class]+=1
        else:
            classVotes[item_class]= 1
    sorted_classVotes =sorted(classVotes.items(), key=lambda kv: kv[1])
    sorted_classVote=list(sorted_classVotes)
    #print("sorted_classVote")
    #print(sorted_classVote)
    return sorted_classVote[-1][0]

"""
build a confusion matrix

input: test elements, predictions
output: confusion matrix

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
""" 
def Confusion_Matrix(test_data,predictions):
    setosa_setosa=0
    setosa_versicolor=0
    setosa_virginica=0
    
    versicolor_setosa=0
    versicolor_versicolor=0
    versicolor_virginica=0
    
    virginica_setosa=0
    virginica_versicolor=0
    virginica_virginica=0
    
    for x in range(len(test_data)):
        acual_label= test_data[x][-1]
        predict_label= predictions[x]
     
        if (acual_label == 'Iris-setosa' and predict_label == 'Iris-setosa'):
            setosa_setosa+=1
        elif (acual_label == 'Iris-versicolor' and predict_label == 'Iris-versicolor'):
            versicolor_versicolor+=1
        elif (acual_label == 'Iris-virginica' and predict_label == 'Iris-virginica'):
            virginica_virginica+=1  
        elif (acual_label == 'Iris-setosa' and predict_label == 'Iris-versicolor'):
            setosa_versicolor+=1
        elif (acual_label == 'Iris-setosa' and predict_label == 'Iris-virginica'):
            setosa_virginica+=1
        elif (acual_label == 'Iris-versicolor' and predict_label == 'Iris-setosa'):
            versicolor_setosa+=1
        elif (acual_label == 'Iris-versicolor' and predict_label == 'Iris-virginica'):
            versicolor_virginica+=1
        elif (acual_label == 'Iris-virginica' and predict_label == 'Iris-setosa'):
            virginica_setosa+=1
        else:
            virginica_versicolor+=1
        
    Confusion_matrix=np.array([setosa_setosa,setosa_versicolor,setosa_virginica,
                               versicolor_setosa,versicolor_versicolor,versicolor_virginica,
                               virginica_setosa,virginica_versicolor,virginica_virginica]).reshape((3,3))
    
    return Confusion_matrix

"""
calculate the accuracy from the confusion matrix

input: confusion matrix
output: return the accuracy

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
""" 
def Accuracy(Confusion_matrix):
    Total_Iris_setosa= Confusion_matrix[0][0]+Confusion_matrix[0][1]+Confusion_matrix[0][2]
    Total_Iris_versicolor= Confusion_matrix[1][0]+Confusion_matrix[1][1]+Confusion_matrix[1][2]
    Total_Iris_virginica= Confusion_matrix[2][0]+Confusion_matrix[2][1]+Confusion_matrix[2][2]
    diagonal= Confusion_matrix[0][0]+Confusion_matrix[1][1]+Confusion_matrix[2][2]
        
    return diagonal/(Total_Iris_setosa+Total_Iris_versicolor+Total_Iris_virginica)
    

"""
KNN algorithm implementation

input: file_input, test data percent, K
output: return the accuracy

Author: Waleed Khalid Bin Salman.
version 1.00 26/9/2020
"""    
def main(file_input,TestData_percent,K_value):
    
    DataSet= loadDataset(file_input)
    Shuffled_DataSet= shuffle_data(DataSet)
    Splited_Shuffled_DataSet= split_data(Shuffled_DataSet,TestData_percent)
    list_Train_Test_DataSet= Train_Test_data(Splited_Shuffled_DataSet)
    groups_accuracy=[]
    accuracy_sum=0
    for group in list_Train_Test_DataSet:
        Predictions=[]
        Train_data= group[1]
        #print(Train_data)
        Test_data= group[0]
        for x in Test_data:
            Neighbors_Dist= Neighbors_dist_L2(Train_data ,x)
            #Neighbors_Dist= Neighbors_dist_L1(Train_data ,x)
            Closest_Neighbors= K_closest_neighbors(Neighbors_Dist,K_value)
            PredictClass= classify_item(Closest_Neighbors)
            Predictions.append(PredictClass)
        
        CM=Confusion_Matrix(Test_data,Predictions)
        accuracy= Accuracy(CM)
        groups_accuracy.append(accuracy)
        
        
    for x in groups_accuracy:
        accuracy_sum+=x
    average_accuracy=accuracy_sum/len(groups_accuracy)
    
    print(average_accuracy) 

# testelement1=[33,5.0,2.77,5.55,2.01,"blalalal"]
# testelement=[32,5.0,2.974,5.55,0.244,"blabla"]
# dataset= loadDataset('Iris.csv')
# Neighbors_Dist=Neighbors_dist_L2(dataset,testelement1)
# Closest_Neighbors=K_closest_neighbors(Neighbors_Dist,1)
# print(Closest_Neighbors)
# PredictClass=classify_item(Closest_Neighbors)
# print(PredictClass)


start_time = time.time()
for x in range(1,9,2):
    print("When K=",x,"The Average Accuracy is= ",end='')
    main('Iris.csv',20,x)    
print("--- %s seconds ---" % (time.time() - start_time))
