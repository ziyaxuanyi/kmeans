import numpy as np

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

if __name__ == "__main__":
    print(img2vector('digits/testDigits/0_1.txt'))