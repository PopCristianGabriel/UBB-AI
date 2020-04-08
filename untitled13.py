import random
from random import choice
from itertools import permutations 
import copy

class Graph:
    
    def __init__(self,problemSize):
        self.problemSize = problemSize
        self.graphs = self.get_all_graphs()
        
        
        
        
        
    def get_graphs(self):
        return self.graphs
    
    def get_problem_sie(self):
        return self.problemSize
    
    def get_trace(self):
        return self.trace
    
    def get_graphs_size(self):
        return len(self.graphs)
    
    
    def get_all_permutaton(self,n):
        pool = permutations([x for x in range(n)],n)
        pool = list(pool)
        return list(pool)
    
    def combine_two_permutations(self,perm1,perm2):
        
        
        permutationForIndivid = []
        for i in range(len(perm1)):
            permutationForIndivid.append([perm1[i],perm2[i]])
        return permutationForIndivid
    
    def get_all_graphs(self):
        graphs = []
        pool = self.get_all_permutaton(self.problemSize)
        for i in range(0,len(pool)):
            for j in range(0,len(pool)):
                graphs.append(self.combine_two_permutations(pool[i],pool[j]))
        return graphs

class Ant:
    def __init__(self, problemSize,initialPath,initialCombination):
        self.problemSize = problemSize
        self.path = [initialPath]
        self.values = [[[-1,-1] for i in range(self.problemSize)] for j in range(self.problemSize)]
        self.values[0][0] = initialCombination
        
    def print_matrix(self):
        for i in range (self.problemSize):
            print(self.values[i])
    def get_values(self):
        return self.values
    
    def dislocate_one_permutation_horizontaly(self,i):
        permutation1 = []
        permutation2 = []
        for k in range(self.problemSize):
            permutation1.append(self.values[i][k][0])
            permutation2.append(self.values[i][k][1])
        return permutation1,permutation2
    
    def dislocate_permutations_horizontaly(self):
        permutations = []
        for i in range(self.problemSize):
            perm1,perm2 = self.dislocate_one_permutation_horizontaly(i)
            permutations.append(perm1)
            permutations.append(perm2)
        return permutations
    
    def discolate_one_permutation_verticaly(self,i):
        permutation1 = []
        permutation2 = []
        for k in range(self.problemSize):
            permutation1.append(self.values[k][i][0])
            permutation2.append(self.values[k][i][1])
        return permutation1,permutation2
    
    def dislocate_permutations_verticaly(self):
        permutations = []
        for i in range(self.problemSize):
            perm1,perm2 = self.discolate_one_permutation_verticaly(i)
            permutations.append(perm1)
            permutations.append(perm2)
        return permutations 
    
    def can_put_new_combunation(self):
        
        matrix = self.values
        for i in range (len(matrix)):
            for j in range(len(matrix)):
                for k in range(len(matrix)):
                    for l in range(len(matrix)):
                        if(i != k or j != l):
                            if(matrix[i][j] == matrix[k][l] and matrix[i][j] != [-1,-1]):
                                return False
                            
        
        collumnPermutations = self.dislocate_permutations_verticaly()
       # linePermutations = self.discolate_permutations_horizontaly()
        linePermutations = self.dislocate_permutations_horizontaly()
        for i in range(len(collumnPermutations)):
            for j in range(len(collumnPermutations[i])):
                if(collumnPermutations[i][j] in collumnPermutations[i][j+1:] and collumnPermutations[i][j] != -1):
                    return False
                if(linePermutations[i][j] in linePermutations[i][j+1:] and linePermutations[i][j] != -1):
                    return False
        return True
    
    def check_new_combination(self,combination):
       for i in range(self.problemSize):
           for j in range(self.problemSize):
               if(self.values[i][j] == [-1,-1]):
                   self.values[i][j] = combination
                   #ok = self.can_put_new_combination()
                   ok = self.can_put_new_combunation()
                   self.values[i][j] = [-1,-1]
                   return ok
    
    def get_next_moves(self,graphs):
       next_moves = []
       for i in range(graphs.get_graphs_size()):
           
           graph = graphs.get_graphs()[i]
           for j in range(len(graph)):
               if(self.check_new_combination(graph[j])):
                   next_moves.append([i,j])
       return next_moves
   
    
    def add_move_based_on_position_in_graph(self,graph,position):
       i = position[0]
       j = position[1]
       for k in range(self.problemSize):
           for l in range(self.problemSize):
               if(self.values[k][l] == [-1,-1]):
                   self.values[k][l] = graph.get_graphs()[i][j]
                   return
               
    def dist_move(self,graph,move):
       dummy = copy.deepcopy(self)
       dummy.path.append(move)
       dummy.add_move_based_on_position_in_graph(graph,move)
       return len(dummy.get_next_moves(graph))
   
    
   
    def fitness(self):
        f = 0
        matrix = self.get_values()
        for i in range (len(matrix)):
            for j in range(len(matrix)):
                for k in range(len(matrix)):
                    for l in range(len(matrix)):
                        if(i != k or j != l):
                            if(matrix[i][j] == matrix[k][l]):
                                f -= -1
        pool = permutations([x for x in range(self.problemSize)],self.problemSize)
        pool = list(pool)
        allPermutations = []
        for i in range(len(pool)):
            allPermutations.append(list(pool[i]))
        
        collumnPermutations = self.dislocate_permutations_verticaly()
        
        for i in range(len(collumnPermutations)):
            if(collumnPermutations[i] not in allPermutations):
                f += 1
        
                                
        return f
   
    def add_move(self,q0, trace, alpha, beta,graphs):
       p = {} 
       nextSteps = self.get_next_moves(graphs)
       if(len(nextSteps) == 0):
           return False
       
       for pozition in nextSteps:
           
           pozition = tuple(pozition)
           p[pozition] = self.dist_move(graphs,pozition)
            
       for key in p:
            i = key[0]
            j = key[1]
            value = p[key]
            value = (value **beta) *(trace[i][j] ** alpha)
            p[key] = value
        
                    
       # p=[ (p[i]**beta)*(trace[self.path[-1]][i]**alpha) for i in range(len(p))]
       if(random.random()<q0):
           
            p2 = []
            for key in p:
                value = p[key]
                p2.append([key,value])
            p2 = max(p2, key=lambda a: a[1])
            self.path.append(p2[0])
            self.add_move_based_on_position_in_graph(graphs, p2[0])
            
       else:
            s = 0
            for value in p.values():
                s += value
            if(s == 0):
                return choice(nextSteps)
            p2 = []
            for key in p:
                p2.append([key,p[key]])
            p2 = [ [p2[i][0],p2[i][1]/s] for i in range(len(p2)) ]
            
            p2 = [ [p2[i][0], sum( [ p2[j][1] for j in range(i+1) ] ) ] for i in range(len(p2)) ]  
           #p2 = [ [p2[i][0],sum(p2[0:i+1][1])] for i in range(len(p2)) ]
            r=random.random()
            i=0
            while (r > p2[i][1]):
                i=i+1
            self.path.append(p2[i][0])
            self.add_move_based_on_position_in_graph(graphs, p2[i][0])
       return True
                

class Problem:
    def __init__(self):
        self.load_data()
        #self.problemSize = problemSize
        #self.nrAnts = nrAnts
        #self.nrIterations = nrIterations
        #self.alpha = alpha
        #self.beta = beta
        #self.q0 = q0
        #self.rho = rho
        self.graphs = Graph(self.problemSize)
        self.trace = [[1 for i in range (self.problemSize)] for j in range(len(self.graphs.get_all_graphs()))]
        self.traceJ = self.problemSize
        self.traceI = len(self.trace)

    def get_random_initial_path_and_combination(self):
        graph = self.graphs.get_graphs()
        i = random.randint(0,len(graph)-1)
        j = random.randint(0,self.problemSize-1)
        return [i,j],graph[i][j]
        
    
    def load_data(self):
        f = open("inp.txt")
        self.problemSize = int(f.readline())
        self.nrAnts = int(f.readline())
        self.nrIterations = int(f.readline())
        self.alpha = float(f.readline())
        self.beta = float(f.readline())
        self.q0 = float(f.readline())
        self.rho = float(f.readline())
        
        
    def iteration(self):
        antSet = []
        for i in range(self.nrAnts):
            pozition,combination = self.get_random_initial_path_and_combination()
            antSet.append(Ant(self.problemSize,pozition,combination))
        for i in range(self.problemSize * self.problemSize):
        # numarul maxim de iteratii intr-o epoca este lungimea solutiei
            for x in antSet:
                x.add_move(self.q0, self.trace, self.alpha, self.beta,self.graphs)
    # actualizam trace-ul cu feromonii lasati de toate furnicile
        try:
            dTrace=[ 1.0 / antSet[i].fitness() for i in range(len(antSet))]
        except ZeroDivisionError:
            return x
        for i in range(self.traceI):
            for j in range (self.traceJ):
                self.trace[i][j] = (1 - self.rho) * self.trace[i][j]
        for i in range(len(antSet)):
            for j in range(len(antSet[i].path)-1):
                x = antSet[i].path[j][0]
                y = antSet[i].path[j][1]
                self.trace[x][y] = self.trace [x][y] + dTrace[i]
        # return best ant path
        f=[ [antSet[i].fitness(), i] for i in range(len(antSet))]
        f=max(f)
        return antSet[f[1]]

    

class Controler:
    
    def __init__(self,problem):
        self.problem = problem
        self.bestSolFitness = 100
        self.bestAnt = 0
        
        
    def solve(self):
        for i in range(self.problem.nrIterations):
            print("epoca : " + str(i))
            bestAnt = self.problem.iteration()
            if(bestAnt.fitness() < self.bestSolFitness):
                self.bestSolFitness = bestAnt.fitness()
                self.bestAnt = bestAnt
                print("fitness = "+str(self.bestSolFitness))
                bestAnt.print_matrix()
                if(self.bestSolFitness == 0):
                    return
            
        
        
    
  
problem = Problem()
controler = Controler(problem)
controler.solve()
        
        
        
        
    
    
    
print("dawd")
    
    