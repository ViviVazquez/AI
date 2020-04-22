"""
This script test the Gradient Descent algorithm for 
Logistics Classification.

Author : Viviana Vazquéz Goméz Martínez
Email : viviana.vazquezgomez@udem.edu
Institution : Universidad de Monterrey
Created : Mon 20 Apr, 2020
"""
import csv
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
#plt.rc("font", size=14)
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from utilityFunction import predict, update_weight_loss, compute_gradient_of_cost_function, compute_L2_norm, eval_hypothesis_function,confusion_matrix, classification_report

def CreateTestTrainSplit(CSVFilename):
    """ 
1. Creates the train and test csv files.

CSVFilename: (str) name of the csv file without the .csv extension
"""    
    from numpy.random import RandomState
    data = pd.read_csv(CSVFilename+".csv")
    rng = RandomState(seed=42) # random seed so that it can consistently produce the same split
    # producing an 80/20 train:test split
    train = data.sample(frac=0.8, random_state=rng) # produces the train split
    test = data.loc[~data.index.isin(train.index)]  # leaves the rest of the info in the test split
    df1 = pd.DataFrame(test)
    df1.to_csv(CSVFilename+"_test.csv") #saves test file csv
    df2 = pd.DataFrame(train)
    df2.to_csv(CSVFilename+"_train.csv") #saves train file csv

    # Test and train data
    Xtest = df1.iloc[:,:-1].values #All row values except the last column (Outcome)
    ytest = df1.iloc[:,-1].values #Label is the last column (Outcome)
    Xtrain = df2.iloc[:,:-1].values#All row values except the last column (Outcome)
    ytrain = df2.iloc[:,-1].values #Label is the last column (Outcome)
    #print(Xtest.shape,Xtrain.shape,ytest.shape,ytrain.shape )
    return Xtrain, Xtest, ytrain, ytest

 

 
def VisualizeData(showPlots=None, CSVFilename=None):
    """
    Visualizing data: creates histograms per class, and violin plots of each class w.r.t the outcome 
    This is for visualizing all the data.  
    """

    plt.rcParams.update({'font.size': 9})
    columns = ['Glucose', 'Age', 'BloodPressure', 'Insulin','BMI','SkinThickness' ,'Pregnancies',  'DiabetesPedigreeFunction']

    if showPlots == True:
        
        nRowsRead = None # specify 'None' if want to read whole file
        df1 = pd.read_csv(CSVFilename+".csv", delimiter=',', nrows = nRowsRead)
        df1.hist( color='pink')
        plt.show()
        
        columns = ['Glucose', 'Age', 'BloodPressure', 'Insulin','BMI','SkinThickness' ,'Pregnancies',  'DiabetesPedigreeFunction']
        n_cols = 2  # 2 figures per plot
        n_rows = 4 
        idx = 0
        df1 = pd.read_csv('diabetes.csv', delimiter=',', nrows = None)

        sns.set(style="whitegrid")
        #plots per number of rows to take into account
        for i in range(n_rows):
            fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(14, 4.3))
            for j in range(n_cols):
                sns.violinplot(x = df1.Outcome, y=df1[columns[idx]], ax=ax[j], color='pink', ) 
                idx += 1
                if idx >= 8:
                    break
        plt.show()


"""
Executes "vivisProject.py" if it is the main
"""

if __name__ == '__main__':
    
    showPlots=False # show (True) or not (False) data visualization
    columns = ['Glucose', 'Age', 'BloodPressure', 'Insulin','BMI','SkinThickness' ,'Pregnancies',  'DiabetesPedigreeFunction']
    CSVFilename = "diabetes" #name of csv file without extension
    
    X_train, X_test, y_train, y_test= CreateTestTrainSplit(CSVFilename)
    #loads the "diabetes.csv" dataset and does a train/test/split

    VisualizeData(showPlots, CSVFilename) #shows histogram plots and violin plots of the complete csv
    
    #creates the first input to be x0=1  (bias initialization)
    intercept = np.ones((X_train.shape[0], 1)) 
    intercept2 = np.ones((X_test.shape[0], 1)) 
    #print(X_test[1].shape)
    # concatenates the bias to the input vector
    x= np.concatenate((intercept, X_train), axis=1)
    x2 = np.concatenate((intercept2, X_test), axis=1)

    O =x[-1,:].shape
    O=list(O)[0]
    w = np.zeros((O,1))
    #print(w.shape)
    """
     Computes the dot product of the weights and the inputs 
     and maps that output onto the sigmoid function (creating the hypothesis).
 
     Then, the difference is computed between the hypothesis and the train label value.
    This is for a quick computation of the gradient of the cost function.
    """

    L3n = 0
    stopping_criteria = 0.01
    learning_rate = 0.0005


    while(True):
        gcf = compute_gradient_of_cost_function(x, y_train, w)


        w =update_weight_loss(w,learning_rate, gcf)
        w=w.T
 
        L2n = compute_L2_norm(w)
        Stop = abs(L2n-L3n)
        

        if Stop < stopping_criteria:
            break

        L3n = L2n
 
    print(" ")
    print("final weights vector", w.T[0])
    print(" ")
    w = w.T[0]
    a = predict(w,x2)


    test_predictions  = (predict(x2, w.T).flatten()>0.5)
    train_predictions = (predict(x, w.T).flatten()>0.5)

    print(" ")
    print("Accuracy on training data: {}".format(accuracy_score(np.expand_dims(y_train, axis=1), train_predictions)))
    print("Accuracy on testing data: {}".format(accuracy_score(np.expand_dims(y_test, axis=1), test_predictions)))
    print(" ")
    y_true= np.expand_dims(y_test, axis=1)

    print("confusion matrix")
    print(confusion_matrix(y_true,test_predictions))
    print(" ")

    print("test classification report")
    print(classification_report(y_true,test_predictions))