
import numpy as np

#we need backprop algorithm 
#Ive set it up already for the second flair, 2 layers each of 100 hidden unit. hiddenLayer[x, y] means that the hiddenlayer list has
#two hidden layers {x, y} and their values are the number of neurons in the layer. 
#We need SGD algorithm 



class MLP:

    def __init__(self, inputs, outputs, hiddenLayers = [1000]) :
        self.inputs = inputs
        self.outputs = outputs
        self.hiddenLayer = hiddenLayers
     

        layers = [] #number of layers we are working with 

       


    def preProcess(self, inputFileName):
        featurematrix = []
        weights = []
        with open(inputFileName, 'r') as file:
            for sentence in file:
                temp = sentence.split()
                featurematrix.append(temp)

        
        featurematrix = np.asanyarray(featurematrix).astype(float) #holds the initial input values

        targetMatrix = featurematrix[:, -1] #holds the target values 
        weights = np.asanyarray(weights)

        weights = np.zeros((np.shape(np.transpose(featurematrix))), dtype=float)


        return featurematrix, targetMatrix, weights
    

    def feedForward(self, inputs, weights):
        x = inputs

        for weight in weights:
            total_input = np.dot(x, weight)

            x = self.tanh(total_input)
        
        return x
    
  

    def tanh(self, a):
        return 2*(1/(1+np.exp(-a)))*(2*(a)) - 1


    #backprop:
    #derivative of tan h is 1-tan^2
    def tan_derivative(output):
        return 1- output**2


# error = (weight[k] * error[j]) * tan_derivative(output)
# error_j is the error signal from the jth neuron in the output layer
# weight_k is the weight that connects the kth neuron to the current neuron
# output is the output for the current neuron.

#neuron number j is also the index of the neuron’s weight in the output layer neuron[‘weights’][j].
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
        #working backwards on output
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
                    #error signal calculated for each neuron is stored with the name ‘delta’.
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * tan_derivative(neuron['output'])

#need to do SGD for training and updating weights
# Update network weights with error
# weight = weight - learning_rate * error * input
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    

def feedForward(inputData, params):
    w1 = params["w1"]
    w2 = params["w2"]
    b1 = params["b1"]
    b2 = params["b2"]

    z1 = np.dot(w1,inputData) + b1
    a1 = tanH(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = softMax(z2)

    feedCache = {"z1": z1, "a1": a1, "z2": z2, "a2":a2}

    return a2, feedCache

def softMax(a):

    return np.exp(a)/np.sum(np.exp(a))

def preProcess(fileName):
    featurematrix = []
    with open(fileName, 'r') as file:
        for sentence in file:
            temp = sentence.split()
            featurematrix.append(temp)
    
    featurematrix = np.asanyarray(featurematrix).astype(float)
    featurematrix = featurematrix[:, :3]
    targetMatrix = np.asanyarray(featurematrix[:, -1]).astype(float)
    targetMatrix = np.reshape(targetMatrix, (targetMatrix.shape[0], 1))

    return targetMatrix, featurematrix


def init(featureMatrix, targetMatrix, hiddenUnits = 100):

    inputUnits = featureMatrix.shape[0]
    outputUnits = 2

    hiddenWeight = np.random.uniform(-1,1,(hiddenUnits, inputUnits))
    hiddenBias = np.zeros((inputUnits, 1))

    outWeight = np.random.uniform(-1,1,(outputUnits, hiddenUnits))
    outBias = np.zeros((outputUnits, 1))



    initParams = {"w1": hiddenWeight, "w2": outWeight, "b1": hiddenBias, "b2":outBias}

    return initParams

def costComp(a2, targets):
    m = targets.shape[0]
    logprobs = np.dot(targets,np.log(a2).T) + np.dot((1-targets),np.log((1-a2)).T)

    cost = -logprobs/m

    return cost

        




 
    


    

