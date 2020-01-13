import operator
import numpy as np

def classify0(inX, dataSet, labels, k):
    """
    params:
    -inX: input vector
    dataSet: train set
    labels: labels vector
    k: k-means neighbor number

    return: result of inX
    """

    dataSetSize = dataSet.shape[0]

    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

if __name__ == "__main__":
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    print(classify0([0, 0], group, labels, 3))