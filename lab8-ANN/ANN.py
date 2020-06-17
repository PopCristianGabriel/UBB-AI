import numpy as np 
import math 

import random
import copy
from datetime import datetime
def activation_sigmoid(x):
    return 1/(1 + np.exp(-x)) 
    

def derivative_sigmoid(y):
    return y*(1-y)

class Data:
    
    def __init__(self,fileLocation,percentage):
        self.fileLocation = fileLocation
        #self.trainingData,self.testData = self.load_data(percentage)
        self.data,self.results,dawd = self.__splitData(self.__normaliseData((self.__loadData2(fileLocation))))
        self.trainingData = self.data[0:(int(percentage/100 * len(self.data)))]
        self.trainingLabels = self.results[0:int((percentage/100 * len(self.results)))]
        self.testData = self.data[int(percentage/100 * len(self.data)):len(self.data)]
        self.testLabels = self.results[int(percentage/100 * len(self.data)) : len(self.results)]
        
        
        
    def get_results(self):
        return self.results

    def __loadData2(self, filename):
      filedata = np.loadtxt(filename, dtype=np.float32)
      return filedata
  
    def __splitData(self, data):
      X = data[:, 0:-1]
      ones = np.ones([X.shape[0], 1])
      X = np.concatenate((ones, X), axis=1)
      y = data[:, [-1]]
      theta = np.zeros([1,len(data[0])])
      return X, y, theta
  
    def __normaliseData(self, data):
      return (data - data.mean()) / data.std()
      
    
    def load_data(self,percentage):
        file = open(self.fileLocation,'r')
        lines = file.readlines()
        testDataPercentage = int(percentage/100*len(lines) / 2)
        testData = [[],[],[],[],[],[]]
        trainData = [[],[],[],[],[],[]]
        i = 0
        for line in lines:
            line = line.split(" ")
            if(len(line) < 2):
                continue
            
            if(i < testDataPercentage):
                for j in range(len(line)):
                    trainData[j].append(float(line[j]))
            else:
                for j in range(len(line)):
                    testData[j].append(float(line[j]))
            i += 1
        return trainData,testData
    

class Matrix:
    
    def __init__(self,nrRows,nrCollums):
        self.nrRows = nrRows
        self.nrCollums = nrCollums
        self.values = [[0 for i in range (nrCollums)] for j in range (nrRows)]
        
    
    def add(self,other):
        result =  Matrix(self.nrRows,self.nrCollums)
        for i in range(self.nrRows):
            for j in range(self.nrCollums):
                result.values[i][j] = self.values[i][j] + other.values[i][j]
                
        return result
    
    def substract(self,other):
        result =  Matrix(self.nrRows,self.nrCollums)
        for i in range(self.nrRows):
            for j in range(self.nrCollums):
                result.values[i][j] = self.values[i][j] - other.values[i][j]
                
        return result
    
    def multiply(self,other):
        result =  Matrix(self.nrRows,other.nrCollums)
        for i in range(self.nrRows):
            for j in range(other.nrCollums):
                suma = 0
                for k in range(other.nrRows):
                    suma = suma + self.values[i][k] + other.values[k][j]
                result.values[i][j] = suma
        return result
    
    
    def multiply_by_constant(self,constant):
        result =  Matrix(self.nrRows,self.nrCollums)
        for i in range(self.nrRows):
            for j in range(self.nrCollums):
                result.values[i][j] = self.values[i][j] * constant
                
        return result
                
    def divide_by_constant(self,constant):
        result =  Matrix(self.nrRows,self.nrCollums)
        for i in range(self.nrRows):
            for j in range(self.nrCollums):
                result[i][j] = self.values[i][j] / constant
                
        return result
    
    
    def add_constant(self,constant):
        result =  Matrix(self.nrRows,self.nrCollums)
        for i in range(self.nrRows):
            for j in range(self.nrCollums):
                result.values[i][j] = self.values[i][j] + constant
                
        return result
    
    def randomize(self):
        for i in range(self.nrRows):
            for j in range(self.nrCollums):
                self.values[i][j] = random.random()
                
                
                
    def apply_function(self,function):
        for i in range(self.nrRows):
            for j in range(self.nrCollums):
                self.values[i][j] = function(self.values[i][j])
    
    def apply_function_static(self,matrix,function):
        result = Matrix(matrix.nrRows,matrix.nrCollums)
        for i in range(matrix.nrRows):
            for j in range(matrix.nrCollums):
                result.values[i][j] = function(matrix.values[i][j])
        return result
    
    def __str__(self):
        s = ''
        for i in range(self.nrRows):
            for j in range(self.nrCollums):
                s += str(self.values[i][j])
    
        return s
    
    def set_matrix(self,value):
        for i in range(self.nrRows):
            for j in range(self.nrCollums):
                self.values[i][j] = value
    
class NeuralNetwork:
    
    def __init__(self,nrInputNodes,nrHiddenNodes,nrOutputNodes,bias,learningRate):
        self.nrInputNodes = nrInputNodes
        self.nrHiddenNodes = nrHiddenNodes
        self.nrOutputNodes = nrOutputNodes
        self.inputBias =  Matrix(self.nrInputNodes,1)
        self.inputBias.set_matrix(bias)
        self.hiddenBias =  Matrix(self.nrInputNodes,1)
        self.hiddenBias.set_matrix(bias)
        self.learningRate = learningRate
        
        self.weights_inputs_hidden =  Matrix(self.nrHiddenNodes,self.nrInputNodes)
        self.weights_hidden_output =  Matrix(self.nrOutputNodes,self.nrHiddenNodes)
        self.weights_hidden_output.randomize()
        self.weights_inputs_hidden.randomize()
        
    
    def fromArrayToMatrix(self,inputs):
        result = Matrix(len(inputs),1)
        
        for i in range(len(inputs)):
            result.values[i][0] = inputs[i]
            
        return result
    
    
    def transposeMatrix(self,matrix):
        result = Matrix(matrix.nrCollums,matrix.nrRows)
        for i in range(matrix.nrRows):
            for j in range(matrix.nrCollums):
                result.values[j][i] = matrix.values[i][j]
        return result
    
    
    def feed_forward(self,inputs):
        
        inputs = self.fromArrayToMatrix(inputs)
        
        hidden = self.weights_inputs_hidden.multiply(inputs)
        hidden = hidden.add(self.inputBias)
        hiddenCopy = copy.deepcopy(hidden)
        hidden.apply_function(activation_sigmoid)
        
        
        output = self.weights_hidden_output.multiply(hidden)
        output.add(self.hiddenBias)
        
        return output,hiddenCopy
    
    
    def train(self,inputs,target):
        #calculate the output with feed forward
        output,hidden = self.feed_forward(inputs)
        #transform the target into a matrix
        target = self.fromArrayToMatrix(target)
        #calculate the error in the output
        error = target.substract(output)
        
        
        #begin backpropagation
        
        #calcualte the transopsed matrix from hidden to output
        weights_hidden_output_transposed = self.transposeMatrix(self.weights_hidden_output)
        #calculate the error in the hidden layer
        hiddenErrors = weights_hidden_output_transposed.multiply(error)
        
        
        gradient = output.apply_function_static(output,derivative_sigmoid)
        gradient.multiply(error)
        gradient.multiply_by_constant(self.learningRate)
        
        hidden_transposed = self.transposeMatrix(hidden)
        weights_hidden_output_deltas = gradient.multiply(hidden_transposed)
        
        self.weights_hidden_output.add(weights_hidden_output_deltas)
        
        #self.inputBias.add(gradient)
        hiddenGradient = output.apply_function_static(hidden,derivative_sigmoid)
        
        hiddenGradient.multiply(self.transposeMatrix(hiddenErrors))
        hiddenGradient.multiply_by_constant(self.learningRate)
        
        
        
        inputsMatrix = self.fromArrayToMatrix(inputs)
        
        weights_inputs_hidden_delta = hiddenGradient.multiply(self.transposeMatrix(inputsMatrix))
        self.weights_inputs_hidden.add(weights_inputs_hidden_delta)
        #self.hiddenBias.add(hiddenGradient)
        
        return target,output,error
        
        
    
        
class Controller:
    
    def __init__(self,data,neuralNetork):
        self.data = data
        self.neuralNetwork = neuralNetork
        self.errorsTraining = []
        self.errorsTesting = []
        
    def solve(self):
        self.train()
    
        self.test()
        
        self.print_results()
    
    def print_results(self):
        avgTrain = 0
        sumTrain = 0
        avgTest = 0
        sumTest = 0
        for i in range(len(self.errorsTraining)):
            sumTrain += float(self.errorsTraining[i][1:len(self.errorsTraining[i])])
        avgTrain = sumTrain / len(self.errorsTraining)
        for i in range(len(self.errorsTesting)):
            q = self.errorsTesting[i][1:len(self.errorsTesting[i])]
            q = q[:-1]
            sumTest += float(q)
        avgTest = sumTest / len(self.errorsTesting)
        
        print("avg error train = ", abs(avgTrain))
        print("avg error test = ",abs(avgTest))
    def train(self):
        trainData = data.trainingData
        trainLabels = data.trainingLabels
        
        for i in range(len(trainData)):
            
            target,output,error = self.neuralNetwork.train(list(trainData[i][1:6]),list(trainLabels[1]))
            self.errorsTraining.append(str(error))
            
    def test(self):
        testData = data.testData
        testLabels = data.testLabels
        
        for i in range(len(testData)):
            target,output,error = self.neuralNetwork.train(testData[i][1:6],[testLabels[1]])
            print("target = ",target)
            print("output = ",output)
            print("error = ",error)
            print("----")
            self.errorsTesting.append(str(error))
        
random.seed(datetime.now())       
data = Data("inputs.txt",70)
nn = NeuralNetwork(5,5,1,0.02,0.01) #nr of imputs, nr of hidden, nr of outputs, bias, learning rate
controller = Controller(data,nn)
controller.solve()
#print(nn.feed_forward([1,0]).values)
        
        
        
        
        
        
        
        