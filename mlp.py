
import numpy as np

#we need backprop algorithm 
#Ive set it up already for the second flair, 2 layers each of 100 hidden unit. hiddenLayer[x, y] means that the hiddenlayer list has
#two hidden layers {x, y} and their values are the number of neurons in the layer. 
#We need SGD algorithm 



class MLP:

    def __init__(self, inputs, outputs, hiddenLayers = [100,100]) :
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




 
    


    

