# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:32:15 2016

@author: okwen

Tutorials from 'Machine Learning in action' by PETER HARRINGTON
http://www2.ift.ulaval.ca/~chaib/IFT-4102-7025/public_html/Fichiers/Machine_Learning_in_Action.pdf4

CLASSIFICATION
"""

from numpy import *
import operator
from os import listdir



def classify0(inX, dataSet, labels, k):
    # Function to run the kNN algorithm on one piece of data
    # For every point in our dataset:
        #calculate the distance between inX and the current point
        #sort the distances in increasing order
        #take k items with lowest distances to inX
        #find the majority class among these items
        #return the majority class as our prediction for the class of inX 


    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#The function classify0() takes four inputs: the input vector to classify called inX,
#our full matrix of training examples called dataSet, a vector of labels called labels,
#and, finally, k, the number of nearest neighbors to use in the voting. The labels vector
#should have as many elements in it as there are rows in the dataSet matrix. You calculate
#the distances B using the Euclidian distance where the distance between two vectors,
#xA and xB, with two elements, is given by:
#    
#    d = ((x*A0-x*b0)**2 + (x*A1-x*b1)**2)**0.5
#
#If we are working with four features, the distance between points (1,0,0,1) and (7, 6, 9, 4)
#would be calculated by
#    ((7-1)**2+(6-0)**2 +(9-0)**2 +(4-1)**2)**0.5

def createDataSet():
    #This creates the dataset and labels
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
    
#    Type the following at the Python command prompt:
#        group,labels = kNN.createDataSet()
#        group


#To predict the class, type the following text at the Python prompt:
#>>> kNN.classify0([0,0], group, labels, 3)


def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#To use this, type the following at the Python prompt:
#>>> reload(kNN)
#>>> datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
#^datingDataMat    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():
#This uses file2matrix and autoNorm() from earlier to get the data into a form you can use. Next, the number of
#test vectors is calculated, and this is used to decide which vectors from normMat will be
#used for testing and which for training. The two parts are then fed into our original
#kNN classifier, classify0. Finally, the error rate is calculated and displayed. Note that
#you’re using the original classifier; you spent most of this section manipulating the
#data so that you could apply it to a simple classifier. Getting solid data is important and
#will be the subject of chapter 20.
#To execute this, reload kNN and then type kNN.datingClassTest() at the Python
#prompt.    
    
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print( "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print( "the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normMat, datingLabels, 3)
    print( "You will probably like this person: %s" % resultList[classifierResult - 1])
    
#kNN.classifyPerson()
    
    #Example: a handwriting recognition system
    
#    Example: using kNN on a handwriting recognition system
#1. Collect: Text file provided.
#2. Prepare: Write a function to convert from the image format to the list format
#used in our classifier, classify0().
#3. Analyze: We’ll look at the prepared data in the Python shell to make sure it’s
#correct.
#4. Train: Doesn’t apply to the kNN algorithm.
#5. Test: Write a function to use some portion of the data as test examples. The
#test examples are classified against the non-test examples. If the predicted
#class doesn’t match the real class, you’ll count that as an error.
#6. Use: Not performed in this example. You could build a complete program to extract
#digits from an image, such a system used to sort the mail in the United States.
    

#There are roughly 200 samples from each digit. The testDigits directory contains about 900
#examples. We’ll use the trainingDigits directory to train our classifier and testDigits to test it

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print( "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print( "\nthe total number of errors is: %d" % errorCount )
    print( "\nthe total error rate is: %f" % (errorCount/float(mTest)))
#you get the contents for the trainingDigits directory B as a list. Then
#you see how many files are in that directory and call this m. Next, you create a training
#matrix with m rows and 1024 columns to hold each image as a single row. You parse out
#the class number from the filename. C The filename is something like 9_45.txt,
#where 9 is the class number and it is the 45th instance of the digit 9. You then put this
#class number in the hwLabels vector and load the image with the function img2vector
#discussed previously. Next, you do something similar for all the files in the testDigits
#directory, but instead of loading them into a big matrix, you test each vector individually
#with our classify0 function. You didn’t use the autoNorm() function from section
#2.2 because all of the values were already between 0 and 1.
    #from python shell type:kNN.handwritingClassTest()