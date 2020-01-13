import numpy as np
from os import listdir

def img2vector(filename):
    """
    param:
    filename: for example:  'digits/testDigits/0_1.txt'

    return: 1*1024 numpyarray
    """
    returnvect = np.zeros((1, 1024))

    f = open(filename)
    for i in range(32):
        line = f.readline()
        for j in range(32):
            returnvect[0, 32*i+j] = int(line[j])
    
    return returnvect

def readHandwritingtTrainSet():
    """
    read trainSet data
    return: trainSet [m, 1024]
            labels
    """
    hwLabels = []

    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)

    trainingMat = np.zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    return trainingMat, hwLabels

if __name__ == "__main__":
    print(img2vector('digits/testDigits/0_1.txt'))
    trainset, label = handwritingClassTest()
    print(trainset[0], label[0])