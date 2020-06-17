import random
import numpy as np
import matplotlib.pyplot as plt
import os
class Data:
    
    def __init__(self,fileLocation,percentage):
        self.fileLocation = fileLocation
        #self.trainingData,self.testData = self.load_data(percentage)
        self.trainingData,self.results,dawd = self.__splitData(self.__normaliseData((self.__loadData2(fileLocation))))
        
        
        
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
    
    def get_ith_train_data(self,i):
        data = []
        for j in range (len(self.trainingData)):
            data.append(self.trainingData[j][i])
        return data
    def get_ith_test_data(self,i):
        return self.testData[i]
    
    
class InputLayer:
    
    
        
    def __init__(self,data):
        self.data = data
        self.weight = random.randrange(0,5)
        
    def get_size(self):
        return len(self.data)
        
    def get_ith_data(self,i):
        return self.data[i]
        
    def get_weight(self):
        return self.weight
        
    def adjust_weight(self,delta):
        self.weight += delta
            
class Results:
    
    def __init__(self,results):
        self.results = results
        
    def get_ith_result(self, i):
        return self.results[i]
    
    
class Perceptron:
    
    def __init__(self,inputLayers,learningRate,results,iterations):
        self.inputLayers = inputLayers
        self.learningRate = learningRate
        self.results = results
        self.iterations = iterations
        self.workingIndex = 0
        
    def activation_function(self,i):
        result = 0 
        errors = 0
        for inputLayer in self.inputLayers:
            x = inputLayer.get_ith_data(i)
            w = inputLayer.get_weight()
            r = self.results.get_ith_result(i)
            result = inputLayer.get_ith_data(i) * inputLayer.get_weight()
            error = self.results.get_ith_result(i) - result
            delta = self.learningRate * error * inputLayer.get_ith_data(i)
            inputLayer.adjust_weight(delta)
            errors += error
        return errors
    
    
    def cls(self):
        os.system('cls' if os.name=='nt' else 'clear')
        
        
    def print_progress(self,j):
        working = ["working (-)","working (\)" , "working (|)","working (/)"]
        if(j % 16 == 0):
                print(chr(27) + "[2J")
                print("")
                print("")
                print("")
                print(working[self.workingIndex%4])
                self.workingIndex += 1
        
    def plot(self,error,iters):
        fig, ax = plt.subplots()
        ax.plot(np.arange(iters), error, 'r')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Mean Error')
        ax.set_title('Error(iteration) plot')
        plt.show()
    
    def solve(self):
        errors = []
        workingIndex = 0
       
        for j in range(self.iterations):
            error = 0
            for i in range(self.inputLayers[0].get_size()):
                error += abs(self.activation_function(i)[0])
            errors.append(error / self.inputLayers[0].get_size())
            self.print_progress(j)
        print(chr(27) + "[2J") 
        print("done! Check plots")
        return errors


class UI:
    
    def __init__(self,fileLocation):
        learningRate = float(input("enter the learning rate:"))
        self.iterations = int(input("enter iterations:"))
        data = Data(fileLocation,100)
        input_layers = [] 
        for i in range(1,6):
            inputLayer = InputLayer(data.get_ith_train_data(i))
            input_layers.append(inputLayer)
            results = Results(data.get_results())
        self.perceptron = Perceptron(input_layers,learningRate,results,self.iterations)
        
        
        
    
    def run(self):
        errors = self.perceptron.solve()
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.iterations), errors, 'r')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
        ax.set_title('Error(iteration) plot')
        plt.show()



def main():
    ui = UI("inputs.txt")
    ui.run()
main()
            
        
            
            