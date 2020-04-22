"""utility libraries for AI"""

import numpy as np
import pandas as pd
import numpy as sqrt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def sigmoid(z):        
    """
    Logistic function for evaluation
    f(z) = 1/(1+exp(-z))
    """
    return 1. / (1 + np.exp(-z))

def predict(inputs,w):  
    """ 
    computes the hypothesis and the sigmoid evaluation
    f(z) = 1/(1+exp(-z))
    where 
    z = w0+ w1 x1 + w2 x2 + ... +w8 x8
    """
    z = np.dot(w,inputs.T)
    a = sigmoid(z)
    return a


def eval_hypothesis_function(w, x):
    """Evaluate the hypothesis function (for training)
    h= W.x
    """
   # print("w shape:",w.T.shape)
   # print("x shape:",x.T.shape)
    z= np.dot(w.T, x.T)
    #print(z.shape)
    return 1. / (1 + np.exp(-z))

def compute_gradient_of_cost_function(x, y, w):
    """computes the gradient of the cost function
    this is given by:  gradient =(output-expected)* input_vector
    """

    # evaluate hypotesis function
    hypothesis_function = eval_hypothesis_function(w, x)
    residual =  np.subtract(hypothesis_function, y)

    gradient_cost_function = np.dot(residual,x)

    return gradient_cost_function


def compute_L2_norm(gradient_of_cost_function):
    """computes the L2-norm on the new weights as
    L2-norm := sqrt( w0^2+w1^2+w3^2+ ... + w8^2+ )
    """
    return np.sqrt(np.sum(np.square(gradient_of_cost_function)))


def update_weight_loss(weight, learning_rate, gradient):
    """
    updates weights as w = w - lr * gradient 
    """
    w = weight.T[0]
    new_gradient = np.multiply(learning_rate,gradient)
    z =  np.subtract(w,new_gradient)
    return z
    
"""
def confusion_matrix(y,Y):
    return confusion_matrix(y,Y)


def classification_report(y,Y):
    return classification_report(y,Y)
"""