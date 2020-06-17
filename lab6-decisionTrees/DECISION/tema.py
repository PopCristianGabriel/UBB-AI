# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:18:40 2020

@author: popcr
"""

import random
import copy

class Tree:
    
    def __init__(self,attributeName,ginis,attributeIndex):
        self.attributeName = attributeName
        self.ginis = ginis
        self.lower = ""
        self.greater = ""
        self.attributeIndex = attributeIndex
        
        
    def get_attribute(self):
        return self.ginis[self.attributeIndex][0]
    
    def get_split(self):
        return self.ginis[self.attributeIndex][1]
    
    def get_distribution(self):
        return self.ginis[self.attributeIndex][2]
    
    def get_value(self):
        return self.ginis[self.attributeIndex][3]
    
    
    def get_zeros(self,i):
        zeros = 0
        d = self.get_distribution()
        for attributeDistribution in  self.get_distribution().values():
            if(attributeDistribution[i]) == 0:
                zeros += 1
        return zeros
    
    def get_final(self,i):
         for className,attributeDistribution in  self.get_distribution().items():
            if(attributeDistribution[i]) != 0:
                className
                
    def get_maximum(self,i):
        maxi = 0
        nameOfClass = "dawd"
        for className,attributeDistribution in  self.get_distribution().items():
            if(attributeDistribution[i]) > maxi:
                nameOfClass= className
                maxi = attributeDistribution[i]
        return nameOfClass
        
        
    def build(self):
        zerosGreater = self.get_zeros(1)
        zerosLower = self.get_zeros(0)
        maxZeros = len(self.get_distribution())
        if(maxZeros-zerosLower == 1):
            self.lower = self.get_final(0)
            if(self.attributeIndex < len(self.ginis)-1):
                self.greater = Tree(self.ginis[self.attributeIndex+1],self.ginis,self.attributeIndex+1)
                self.greater.build()
                return
            else:
                self.greater = self.get_maximum(1)
                
        if(maxZeros-zerosGreater == 1):
            self.greater = self.get_final(1)
            if(self.attributeIndex < len(self.ginis)-1):
                self.lower = Tree(self.ginis[self.attributeIndex+1],self.ginis,self.attributeIndex+1)
                self.lower.build()
                return
            else:
                self.greater = self.get_maximum(0)
                
        self.lower = self.get_maximum(0)
        self.greater = self.get_maximum(1)
        
        
    def parse(self,individ):
        copyTree = copy.deepcopy(self)
        while isinstance(copyTree , Tree):
            if(individ[self.attributeName] < copyTree.ginis[self.attributeIndex][1]):
                copyTree = copyTree.lower
                
            else:
                copyTree = copyTree.greater
                
        return copyTree
        
            
            
        
        
            
            

class Problem:
    def __init__(self,data):
        self.data = data
        
    def calculate_gini_indexes(self):
        ginis = []
        for i in range(1,self.data.nrOfAttributes):
            
            random_split = random.randint(1,self.data.max[i]-1)
            distribution = {}
            for j in range(len(self.data.data[i])):
                attributeValue = self.data.data[i][j]
                classValue = self.data.data[0][j]
                if classValue in  distribution:
                    if(attributeValue <= random_split):
                        distribution[classValue] =[distribution[classValue][0]+1,distribution[classValue][1]]
                    else:
                        distribution[classValue] =[distribution[classValue][0],distribution[classValue][1]+1]
                        
                else:
                    if(attributeValue <= random_split):
                        distribution[classValue] =[1,0]
                    else:
                        distribution[classValue] =[0,1]
                        
          
            weightGreater = 1
            weightLower = 1
            sumaGreater = sum(attributeDistribution[1] for attributeDistribution in distribution.values())
            sumaLower = sum(attributeDistribution[0] for attributeDistribution in distribution.values())
            for attributeDistribution in distribution.values():
                suma = attributeDistribution[0] + attributeDistribution[1]
                
                weightGreater -= ((attributeDistribution[1]/sumaGreater)**2)
                weightLower -= ((attributeDistribution[0]/sumaLower)**2)
            
            gini = 0
            sumGreater = 0
            sumLower = 0
            for attributeDistribution in distribution.values():
                sumGreater += (attributeDistribution[1])/self.data.dataSetSize
                sumLower += (attributeDistribution[0])/self.data.dataSetSize
                
            gini = sumLower * weightLower + sumGreater * weightGreater
            ginis.append([i,random_split,distribution,gini])
        return ginis
    
    
    def construct_tree(self):
        ginis = self.calculate_gini_indexes()
        ginis = sorted(ginis,key=lambda x: x[3])
        self.tree = Tree(ginis[0][0],ginis,0)
        self.tree.build()
    
    def clasify(self):
        correct = 0
        results = []
        for i in range(self.data.testDataSize):
            test = []
            
            for j in range (self.data.nrOfAttributes):
                test.append(self.data.testData[j][i])
            result = self.tree.parse(test)
            results.append([test[0],result])
         #   print(test[0] +" " + result)
            if(test[0] == result):
                correct += 1
        #print("accuracy = "+str(correct / len(self.data.testData[0])))
        return correct / len(self.data.testData[0]),results
            
    def solve(self):
        self.construct_tree()
        return self.clasify()
        
    
                
                



class Data:
    def __init__(self,filepath,percentageTrain):
        self.data = []
        self.filepath = filepath
        self.max = ['NuConteazaCeEAici:(']
        self.testData = []
        self.percentageTrain = percentageTrain
        self.load_data()
        self.set_maximum_for_attributes()
        
        
        
        
    def set_nr_of_data_attributes(self,nrOfAttributes):
        for i in range(nrOfAttributes):
            self.data.append([])
            self.testData.append([])
        self.nrOfAttributes = nrOfAttributes
        
    def get_attribute_list(self,i):
        return self.data[i]
        
    def load_data(self):
        
        f = open(self.filepath,'r')
        lines = f.readlines()
        
        self.set_nr_of_data_attributes(len(lines[0].split(',')))
        count = 0
        trainDataNr = int(self.percentageTrain/100*len(lines))
        self.dataSetSize = trainDataNr
        self.testDataSize = len(lines)-trainDataNr
        for line in lines:
            line = line.split(',')
            if(count < trainDataNr):
                for i in range(len(line)):
                    try:
                        self.data[i].append(int(line[i]))
                    except ValueError:
                        self.data[i].append(line[i])
            else:
                for i in range(len(line)):
                    try:
                        self.testData[i].append(int(line[i]))
                    except ValueError:
                        self.testData[i].append(line[i])
            count += 1
    
    def set_maximum_for_attributes(self):
        for i in range(1,self.nrOfAttributes):
            maxi = 0
            for j in range(len(self.data[i])):
                if(self.data[i][j] > maxi):
                    maxi = self.data[i][j]
            self.max.append(maxi)
            
def main():                
    maxi = 0
    
    iterations = int(input("enter nr of iterations:"))
    trainPercentage = int(input("enter the train percentage: (ex '80'  means 80%)\n"))
    
    data = Data("balance-scale.data",trainPercentage)
    random.seed(a=None, version=2)  
    bestResults = []
    for i in range(iterations):
        
        problem = Problem(data)
        acc,results = problem.solve()
        if(maxi < acc):
            maxi = acc
            bestResults = results
        if(i%400 == 0):
            print("iteration "+str(i) + " best accuracy:" + str(maxi))
            print("best accuracy's results: (left is expected result, right is prediction)")
            print(bestResults)
        
    print(maxi)
    
main()