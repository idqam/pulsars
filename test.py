import numpy as np
def preProcess(inputFileName):
        #data should be a file name
        featurematrix = []
        with open(inputFileName, 'r') as file:
            for sentence in file:
                temp = sentence.split()
                featurematrix.append(temp)

            

        
        featurematrix = np.asanyarray(featurematrix).astype(float) #100 cols and 100 rows

        targetMatrix = featurematrix[:, -1] #holds the target values 1 col 100 rows

        print(targetMatrix)



def main():

    preProcess("linearSmoke")

if __name__ == "__main__":
    main()