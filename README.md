# AI
artificial intelligence class projects


Python script that pseudo-randomly splits the 'diabetes.csv' dataset
into a training set: 'diabetesTrain.csv' and a test set: 'diabetesTest.csv'

Then, proceeds to do the gradient descent method with a sigmoid activation function
in order to minimize the cost or log-likelihood function.

There is an L2-norm criteria for stopping the classification.
If the difference between the old and new weight vector norm is
less than 0.01, then the iterations stop calculating the gradient descent.

At the end, predictions are made with the test set giving 
a full confusion matrix and classification report.

If the output prediction is > 0.5 (mapping on the sigmoid function), 
then we classify the person as diabetic. 
And if the output prediction is < 0.5 we predict the person is not diabetic.

Confusion matrix shows that we have good oportunities of predicting when a person is 
not diabetic, but almost null chances of predicting when a person is diabetic.

The final weights vector gives us an inside on how insulin and skin thickness
are the most influential factors for a person who may be diabetic.
