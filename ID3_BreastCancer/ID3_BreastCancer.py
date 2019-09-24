#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import os
from collections import Counter

training_data = []
test_data = []

def PreprocessData():
    dictKeys = ['ID', 'ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape',
                'MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei', 
                'BlandChromatin', 'NormalNucleoli', 'Mitoses', 'Class']
    file = open("breast-cancer-wisconsin.data", "r")
    PREtraining_data = file.read()
    entry = ((dict()), str())

    entryDict = dict()
    entryClass = str()

    currentValue = str()
    currentKey = 0
    currentEntry = 0

    for ch in PREtraining_data:
        if ch != ',' and ch != '\n':
            currentValue += ch
        elif ch == ',':
            if currentValue != '?':
                currentValue = int(currentValue)
            if dictKeys[currentKey] != 'ID':
                entryDict[dictKeys[currentKey]] = currentValue
            currentValue = str()
            currentKey += 1
        elif ch == '\n':
            if currentValue != '?':
                currentValue = int(currentValue)
            if currentValue == 2:
                entryClass = 'benign'
            elif currentValue == 4:
                entryClass = 'malignant'
            currentValue = str()
            entry = (entryDict, entryClass)
            if currentEntry < 500:
                training_data.append(entry)
            else:
                test_data.append(entry)
            currentKey = 0
            currentEntry += 1
    return


def CalcNumOfEntries(dataSet):
    return len(dataSet)

def CalcClassInfo(dataSet):
    possible_labels = []
    for input, class_label in dataSet:
        possible_labels.append(class_label)

    possibleClasses = set(possible_labels)
    numClasses = len(possibleClasses)
    return (possibleClasses, numClasses, dict(Counter(possible_labels)))

def Entropy(dataSet):
    numOfEntries = CalcNumOfEntries(dataSet)
    classInfo = CalcClassInfo(dataSet)
    possibleClasses = classInfo[0]
    numOfClasses = classInfo[1]
    frequency = classInfo[2]

    amountOfEntropy = 0
    for Cj in frequency:
        value = frequency.get(Cj)
        amountOfEntropy -= (value / numOfEntries) * np.log2(value / numOfEntries)
    return amountOfEntropy

def EntropyAi(dataSet, selectedAttribute):
    amountOfEntropy = 0
    numOfEntriesD = CalcNumOfEntries(dataSet)
    dataSets = []
    attClasses = []

    for input, classLabel in dataSet:
        for att in input:
            if att == selectedAttribute:
                attClasses.append(input.get(att))

    attClasses = set(attClasses)
 
    iterator = 0
    while iterator < len(attClasses):
        dataSets.append([])
        iterator += 1

    iterator = 0
    for C in attClasses:
        for input, classLabel in dataSet:
            for att in input:
                if att == selectedAttribute and input.get(att) == C:
                    dataSets[iterator].append((input, classLabel))
        iterator += 1

    for dataJ in dataSets:
        amountOfEntropy += (CalcNumOfEntries(dataJ) / numOfEntriesD) * Entropy(dataJ)

    return amountOfEntropy

def CalcInformationGain(dataSet, selectedAttribute):
    amountOfGain = Entropy(dataSet) - EntropyAi(dataSet, selectedAttribute)
    return amountOfGain

def Generate_DT(dataSet, attributeList):
    classInfo = CalcClassInfo(dataSet)
    possibleClasses = classInfo[0]
    numOfClasses = classInfo[1]
    frequency = classInfo[2]

    majorityClass = None
    for C in frequency:
        if majorityClass == None or frequency.get(C) > frequency.get(majorityClass):
            majorityClass = C

    # ID3 algorithm starts here    
    # If samples are all of the same class C:
    if numOfClasses == 1:
        # Then return the current node as a leaf node labeled with class C
        for c in possibleClasses:
            return c

    # If the attribute list is empty:
    elif not attributeList:
        # Then return the current node as a leaf node labeled with the majority class
        return majorityClass

    # Otherwise, it's not a leaf node, so do some magic.
    else:
        maxGain = (None, 0)
        attributeGainInfo = (None, 0)

        for att in attributeList:
            attributeGainInfo = (att, CalcInformationGain(dataSet, att))
            if attributeGainInfo[1] >= maxGain[1]:
                maxGain = attributeGainInfo

        selectedAttribute = maxGain[0]
        
        attributeClasses = []
        for input, class_label in dataSet:
            for attribute in input:
                if attribute == selectedAttribute:
                    attributeClasses.append(input.get(attribute))
        attributeClasses = set(attributeClasses)

        theBranches = dict()
        # For each known value C of selectedAttribute
        for att in attributeClasses:
            newDataSet = []
            for input, classLabel in dataSet:
                for att2 in input:
                    if att2 == selectedAttribute and input.get(att2) == att:
                        newDataSet.append((input, classLabel))

            if not newDataSet:
                theBranches[att] = majorityClass
                
            else:
                newAttributeList = attributeList.copy()
                newAttributeList.remove(selectedAttribute)
                theBranches[att] = Generate_DT(newDataSet, newAttributeList)

        theBranches[None] = majorityClass
        return (selectedAttribute, theBranches)

def main():
    PreprocessData()

    attributeList = set(training_data[0][0].keys())
    theTree = Generate_DT(training_data, attributeList)
    print("Here's the determined Decision Tree:")
    print("{}".format(theTree))

    print(); print(); print()

    Correct = 0
    Wrong = 0

    for dataDict, dataClass in test_data:
        if dataClass == Classify(theTree, dataDict):
            Correct += 1
        else:
            Wrong += 1

    print("Correct: " + str(Correct))
    print("Wrong: " + str(Wrong))


def Classify(theTree, theSample):
    if theTree == 'benign' or theTree == 'malignant':
        return theTree
    else:
        for att in theTree:
            for att2 in theSample:
                if att == att2 and theTree[1].get(theSample.get(att2)) != None:
                    return Classify(theTree[1].get(theSample.get(att2)), theSample)

        return Classify(theTree[1].get(None), theSample)

main()


