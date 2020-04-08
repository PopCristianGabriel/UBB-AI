# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:00:25 2020

@author: Pop
"""

import copy
class Configuration:
    
    def __init__(self,positions,n):
        self.n = n
        self.size = len(positions)
        self.positions = positions
        
        
    def is_allowed(self):
        
       if(self.size > self.n):
            return False
       for i in range(self.size-1):
           for j in range(i+1,self.size):
               if(self.positions[i][0] == self.positions[j][0]):
                   return False
               if(self.positions[i][1] == self.positions[j][1]):
                   return False
               if(abs(self.positions[i][0] - self.positions[j][0]) - abs(self.positions[i][1] - self.positions[j][1]) == 0):
                   return False
       return True
   
    
    def get_size(self):
        return self.size
    
    def get_positions(self):
        return self.positions
    
    
    def getSize(self):
        return self.size
    
    def getValues(self):
        return self.positions[:]
    
    def already_in(self,pair):
        return pair in self.positions
        
    
    def add_queen(self,pair):
        self.positions.append(pair)
    
    def next_config(self):
        nextC = []
        
        for i in range(self.n):
            for j in range(self.n):
                pair = [i,j]
                
                if(not self.already_in(pair)):
                    newConfiguration = copy.deepcopy(self)
                    newConfiguration.add_queen(pair)
                    if(self.is_allowed() == True):
                        newConfiguration.size += 1
                        nextC.append(newConfiguration)
        return nextC
    
    
    def __eq__(self, other):
        if not isinstance(other, Configuration):
            return False
        if self.size != other.getSize():
            return False
        for i in range(self.size):
            if self.values[i] != other.getValues()[i]:
                return False
        return True
    
    def __str__(self):
        List = []
        for pair in self.positions:
            List.append(pair)
        return str(List)
    
class State:
    '''
    holds a PATH of configurations
    '''
    def __init__(self):
        self.values = []
    
    def setValues(self, values):
        self.values = values[:]

    def getValues(self):
        return self.values[:]

    def __str__(self):
        s=''
        for x in self.values:
            s+=str(x)+"\n"
        return s

    def __add__(self, something):
        aux = State()
        if isinstance(something, State):
            aux.setValues(self.values+something.getValues())
        elif isinstance(something, Configuration):
            aux.setValues(self.values+[something])
        else:
            aux.setValues(self.values)
        return aux
    

class Problem:
    def __init__(self,initialConfig,finalConfig):
        self.initialConfig = initialConfig
        self.finalConfig = finalConfig
        self.initialState = State()
        self.initialState.setValues([initialConfig])
        
    def heuristics(self,state):
        lastConfig = state.getValues()[-1]
        count = self.finalConfig.get_size() + lastConfig.getSize()
        for pair in lastConfig.get_positions():
            if(pair not in self.finalConfig.get_positions()):
                count -= 1
        return count
    

    def expand(self, currentState):
      
        myList = []
        currentConfig = currentState.getValues()[-1]
        for j in range(currentConfig.getSize()):
            for x in currentConfig.next_config():
                
                myList.append(currentState+x)
        
        return myList
        
    
    
   
        
    
    
    def getFinal(self):
        return self.finalConfig
    
    def getRoot(self):
        return self.initialState
            
                    
    
    

class Controller:
    def __init__(self,problem):
        self.problem = problem
        
        
    
    def end_BestFS(self,node):
        
        if(len(node.getValues()[-1].positions) != len(self.problem.getFinal().positions)):
            return False
        for pair in node.getValues()[-1].positions:
            if(pair not in self.problem.getFinal().positions):
                return False
            
        return True
    
    
    
    
    def DFS(self, root):
        
        q = [root]

        while len(q) > 0 :
            currentState = q.pop(0)
            
            if self.end_BestFS(currentState):
                return currentState
            q = q + self.problem.expand(currentState)
    
    def Greedy(self, root):
        
        visited = []
        toVisit = [root]
        z = 0
        while len(toVisit) > 0:
            z += 1
            
            node = toVisit.pop(0)
            visited = visited + [node]
            a = node.getValues()
            b = self.problem.getFinal()
            c = a[0]
           
            #if([0,1] in node.getValues()[-1].positions and [1,3] in node.getValues()[-1].positions):
           # print(str(node.getValues()[-1].positions) + " - " + str((self.problem.getFinal().positions)))
            if self.end_BestFS(node):
                return node
            aux = []
            for x in self.problem.expand(node):
                if x not in visited:
                    aux.append(x)
            aux = [ [x, self.problem.heuristics(x)] for x in aux]
            aux.sort(key=lambda x:x[1])
            aux = [x[0] for x in aux]
            toVisit = aux[:] + toVisit     
   
                


class UI:
    
    def __init__(self,controller):
        
        self.controller = controller
        
    
   
    
    def main_menu(self):
        print("1-gbfs")
        print("2-dfs/bfs")
        
        choice = int(input("enter a number:"))
        if(choice == 1) :
            print(str(self.controller.Greedy(self.controller.problem.getRoot())))
        else:
            print(str(self.controller.DFS(self.controller.problem.getRoot())))
        print("dawd")
        
def read_from_file():
    file = open("nqueen.txt", "r")
    n = int(file.read(1))
    q = file.read(1)
    line = file.readline()
    line = line.split(' ')
    
    
    line2 = []
    pair = []
    for i in range (len(line)):
        if(len(pair) == 2):
            line2.append(copy.deepcopy(pair))
            pair.clear()
            pair.append(int(line[i]))
        else:
            pair.append(int(line[i]))
    
    
    line2.append(pair)
   
       
    initialConfig = Configuration(line2,n)
    line = file.readline()
    line = line.split(' ')
    line2 = []
    pair = []
    
    for i in range (len(line)):
        if(len(pair) == 2):
            line2.append(copy.deepcopy(pair))
            pair.clear()
            pair.append(int(line[i]))
        else:
            pair.append(int(line[i]))
    line2.append(pair)
    finalConfig = Configuration(line2,n)
    file.close()
    return initialConfig,finalConfig
    
def run():
    initialConfig,finalConfig = read_from_file()
    problem = Problem(initialConfig,finalConfig)
    controller = Controller(problem)
    ui = UI(controller)
    ui.main_menu()
        
run()