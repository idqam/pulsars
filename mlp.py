
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
     

        layers = [100,100] #number of layers we are working with 

        self.weights = [] #list of weight matrices 
        
        for n in range(len(layers) - 1 ):
            w = np.zeros(layers[n], layers[n+1] ); #creates a matrix filled with zeros
            self.weights.append(w)



    def preProcess(self, inputData):
        #data should be a file name
        nameofFile = input("Enter the name of the file: ")
        file = open(inputData, 'r')
        inputsList = []
        for line in file:
            tempArr = line.split()
            inputsList.append(tempArr)
        
        featurematrix = np.asanyarray(inputsList).astype(float)

        targetMatrix = np.delete(featurematrix, [0,1,2], 1) #holds the target values

        return featurematrix, targetMatrix
    

    def feedForward(self, inputs):
        x = inputs

        for weight in self.weights:
            total_input = np.dot(x, weight)

            x = self.tanh(total_input)
        
        return x
    
  

    def tanh(self, a):
        return 2*(1/(1+np.exp(-a)))*(2*(a)) - 1




 
    


    

