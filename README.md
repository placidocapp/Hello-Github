# Hello-Github
First time in Github

This was my first neural network, fully implemented by myself. 

The dataset can be found here http://yann.lecun.com/exdb/mnist/, 
I just read de data and saved into X_train, X_test, Y_train and
Y_test.

The functions:
sigmoid --> Implements a sigmoid elementwise.

costFunction --> Implements a foward propagation, calculate the 
cost function of a logistic regression and then back propagation
to obtain the derivatives.

Gradient_verification --> Insite this function there is a parameter
that one could change, that parameter change how many thetas 
derivatives the function will compare. Here I just use some small
eps and usa a numerical aproximation of the derivative of each theta
by the definition of derivative. (I recomend compare just some thetas,
because thats enough and it takes loong time to calculate each derivative
with this function).

cost -> Just do foward propagation and calculate the cost, its just a part
of the costFunction.

NeuralNetwork --> Here I put everything together, the gradient checking is 
commented, to test it just uncomment. The function save the theta values
after the gradient descent and in the end it plots the train accuracy and 
the test accuracy.


