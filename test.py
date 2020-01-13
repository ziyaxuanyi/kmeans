from os import listdir
from data import img2vector, readHandwritingtTrainSet
from model import classify0

if __name__ == "__main__":
    trainingMat, hwLabels = readHandwritingtTrainSet()
    testFileList = listdir('digits/testDigits')

    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)

        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        print("test sample %d, pred: %d, ground true: %d" % (i+1, classifierResult, classNumStr))

        if(classifierResult != classNumStr):
            errorCount += 1.0
    
    print("\nerror number count: %d" % errorCount)
    print("\nerror rate: %f" % (errorCount/float(mTest)))