import numpy as np
def preProcess(inputFileName):
        #data should be a file name
        featurematrix = [] 
        weights =[]
        with open(inputFileName, 'r') as file:
            for sentence in file:
                temp = sentence.split()
                featurematrix.append(temp)

            

        
        featurematrix = np.asanyarray(featurematrix).astype(float) #100 cols and 100 rows

        weights = np.asanyarray(weights)

        weights = np.zeros((np.shape(np.transpose(featurematrix))), dtype=float)

        targetMatrix = featurematrix[:, -1] #holds the target values 1 col 100 rows
        print(np.shape(featurematrix))
        print(np.shape(weights))
        print(weights)



        #print(targetMatrix)



def main():

    preProcess("linearSmoke")

if __name__ == "__main__":
    main()