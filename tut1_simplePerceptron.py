import numpy as np
import matplotlib.pyplot as plt

def drawLine(x1, x2):
    ln = plt.plot(x1, x2, '-')
    plt.pause(0.01) # pause between frame (in a loop)
    ln[0].remove() # delete current line to draw next line (better model)


# signmoid activation function 
# this function calculate the "class" of the "weighted sum"
# this "class" ranging from class "0" --> class "1": 0 ... 0.5 ... 1
def sigmoid(weightedsum): 
    return 1/(1 + np.exp(-weightedsum))

# this is the simplest type of feedforward
def guess (line_params, points):
    # input is ARRAY of points
    # an ARRAY of weighted sum of each point :)
    # x1*w1 + x2*w2 + 1*b = "weighted sum"
    weightedsums = points * line_params

    # output is ARRAY of propability of each point 
    p = sigmoid(weightedsums)
    
    return p


# this is the simpest type of back propagation
# training the the model using gradient descent
def train (line_parameters, points, labels):
    learningRate = 0.09
    total_n_points = points.shape[0]  # total rows are just number of points
    preds = guess(line_parameters, points)
    
    # gradient descent final fomular (previously proved) if ...
    # ... minimize mean squared error function 
    # or cross entropy error function ... hmmm ... leave this for later
    # calculate gradient (derivative)
    gradient = points.T * (preds - labels)*(1 / total_n_points)* learningRate

    # adjust the params of the line (model) (adjust the weight) 
    line_parameters = line_parameters - gradient 
    return line_parameters # return a new trained model


# calculate error comes out of perceptron
# using cross entropy error function
def calculateError(line_params, points, labels):
    pred = guess(line_params, points) # predict == array of propability 
    total_points = points.shape[0] # total number of rows
    # mean(-sum( log(p)*y + log(1 - p) * (1 - y) ))
    # we need to transpose ".T" to perform dot product of "propability" and "labels" 
    # (this perform matrix math - on 2d array: rows and cols)
    cross_entropy_error = -(1/total_points) * (np.log(pred).T * labels + np.log(1 - pred).T * (1 - labels))

    # the larger the error, the worse the model (line parameter) performs
    return cross_entropy_error[0, 0]


# ---------------------- preparing DATAS -------------------------------------------

n_pts = 10 # number of points
bias = np.ones(n_pts) # bias input

random_points_top_region = np.array([
    # normal distribution: mean (centre value), std deviation (spread), size of the array
    np.random.normal(10, 2, n_pts), # random points x
    np.random.normal(12, 2, n_pts),  # random points y
    bias # there are 3 inputs into the perceptron: x1, x2(or y) and bias = 1
]).T # transpose

random_points_bottom_region = np.array([
    np.random.normal(5, 2, n_pts), # random points x
    np.random.normal(6, 2, n_pts),  # random points y
    bias # there are 3 inputs into the perceptron: x1, x2(or y) and bias = 1
]).T # transpose

# we want to combine both top regions array and bottom regions into one array vertically, 
# it means on top of each other

all_points = np.vstack((random_points_top_region, random_points_bottom_region))

# we need labels, depends on our design
# the first 10 poinsts should above the line (the final desired model), --> label 0
# the last 10 poinsts should below the line (the final desired model) --> label 1
labels = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)

# --------------------------------------****-----------------------------------------

# ------------------------------- LINE EQUATION (The model) -------------------------------------

# display some random line on the plot
# the equation for a line is:
# or it is called a LINEAR MODEL
# y = a*x + b 
# --> called x2 = y, x1 = x
# --> x2 = a*x1 + b
# --> a*x1 - x2 + b = 0
# the "= 0" happends when a point (x1,x2) is on that line

# but for simple classification, some points (x1, x2) might be above or below the line
# so the a*x1 - x2 + b = "some value" 
# "some value" will indicate whether a point is above or below that line 

# and we can rewrite as:
# --> w1*x1 + w2*x2 + b = "some value" == "weighted sum"
# --> so we just trying to CONVERT THE LINE PARAMS INTO WEIGHTS of the perceptron
# or I'd say TRAINSFORM LINEAR MODEL INTO A PERCEPTRON 
# a LINEAR MODEL (with 2 inputs x1, x2) is a ... 
# ... PERCEPTRON with 2 params (w1, w2) and one output (for classification) 
# inputs are features of the perceptron
# also, the output "some value" is called "weighted sum" in perceptron

# to draw that equation as a line: 
# --> w1*x1 + w2*x2 + b = 0 
# --> x2 = (-w1/w2) * x1 + (-1/w2) * b

# init random weights for parameters of a line
w1 = 0 
w2 = 0
b = 0 # bias weight

# simply a 2d arrays but with matrix math that we'll make use of
line_params = np.matrix([w1, w2, b])
# must transpose to perform several matrix math
line_params = line_params.transpose()  

# --------------------------------------****--------------------------------------------



# draw the datas 
_, ax = plt.subplots(figsize=(4, 4)) # figsize = 4x4 inches
ax.scatter(random_points_top_region[:,0], random_points_top_region[:, 1], color='r')
ax.scatter(random_points_bottom_region[:,0], random_points_bottom_region[:, 1], color='b')

# training the model in a loop
for i in range(2000):
    line_params = train(line_params, all_points, labels)
    print('error: ', calculateError(line_params, all_points, labels)) 

    # get the params after training
    w1 = line_params.item(0)
    w2 = line_params.item(1)
    b = line_params.item(2)

    # create start point and end point to draw line
    # we get left most x and right most x
    two_points_x1 = np.array([
        random_points_bottom_region[:,0].min(), 
        random_points_top_region[:,0].max()
    ]) 

    # calculate y (or x2) - just put it in a math calculation 
    # --> element wise (go though each element in x1)
    # --> result x2 will be an array of 2 elements:
    # y (or x2) of left most x
    # y (or x2) of right most x
    two_points_x2 = (-w1/w2) * two_points_x1 + (-1/w2) * b

    # show the line
    drawLine(two_points_x1, two_points_x2)

plt.show()




