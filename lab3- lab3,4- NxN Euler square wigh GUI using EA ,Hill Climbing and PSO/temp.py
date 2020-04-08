
import random
from itertools import permutations 
import copy
import sys
import time
import threading
import _thread
from qtpy.QtWidgets import (QWidget, QPushButton, QLineEdit,QLabel, 
    QInputDialog, QApplication,QVBoxLayout)
from qtpy.QtCore import *
from qtpy.QtGui import *
class Individ:
    
    
    
    
    
    def __init__(self,size):
        self.size = size
        self.values = []
        for i in range(size):
            self.values.append(self.combine_two_permutations(self.size))
            
    def get_values(self):
        return self.values
    
    def __str__(self):
        return self.values
    def print_matrix(self):
        for i in range (self.size):
            print(self.values[i])
    
    def set_values(self,values):
        self.values = values
    
    def dislocate_one_permutation_horizontaly(self,i):
        permutation1 = []
        permutation2 = []
        for k in range(self.size):
            permutation1.append(self.values[i][k][0])
            permutation2.append(self.values[i][k][1])
        return permutation1,permutation2
    
    def dislocate_permutations_horizontaly(self):
        permutations = []
        for i in range(self.size):
            perm1,perm2 = self.dislocate_one_permutation_horizontaly(i)
            permutations.append(perm1)
            permutations.append(perm2)
        return permutations
    
    def discolate_one_permutation_verticaly(self,i):
        permutation1 = []
        permutation2 = []
        for k in range(self.size):
            permutation1.append(self.values[k][i][0])
            permutation2.append(self.values[k][i][1])
        return permutation1,permutation2
    
    def dislocate_permutations_verticaly(self):
        permutations = []
        for i in range(self.size):
            perm1,perm2 = self.discolate_one_permutation_verticaly(i)
            permutations.append(perm1)
            permutations.append(perm2)
        return permutations
    
    def get_random_permutaton(self,n):
        pool = permutations([x for x in range(n)],n)
        pool = list(pool)
        return list(pool[random.randint(0,len(pool)-1)])
        
        
        
    def combine_two_permutations(self,n):
        perm1 = self.get_random_permutaton(self.size)
        perm2 = self.get_random_permutaton(self.size)
        
        permutationForIndivid = []
        for i in range(len(perm1)):
            permutationForIndivid.append([perm1[i],perm2[i]])
        return permutationForIndivid
    
    def check_if_can_put_perm(self,permutationToAdd):
        for i in range(self.size):
            if(permutationToAdd == self.values[i]):
                return False
        return True
        
    def get_size(self):
        return self.size
    

    def set_matrix(self,permutations):
        matrix = []
        i = 0
        while(i < (len(permutations)-1)):
        
            row = []
            for j in range(self.size):
                cell = [permutations[i][j],permutations[i+1][j]]
                row.append(cell)
            matrix.append(row)
            i += 2
        self.set_values(matrix)
        
    
class Population:
    
    def __init__(self,count,size):
        self.count = count
        self.individuals = []
        for i in range(self.count):
            self.individuals.append(Individ(size))
            
    def get_count(self):
        return self.count
        
    def get_population(self):
        return self.individuals
    


class Problem:
    
    def __init__(self,count,size,chanceForMutation,iterations):
        self.problem = Population(count,size)
        self.count = count
        self.size = size
        self.chanceForMutation = chanceForMutation
        self.iterations = iterations
        self.iterationNr = 0
        
        
    def fitness(self,individ):
        f = 0
        matrix = individ.get_values()
        for i in range (len(matrix)):
            for j in range(len(matrix)):
                for k in range(len(matrix)):
                    for l in range(len(matrix)):
                        if(i != k or j != l):
                            if(matrix[i][j] == matrix[k][l]):
                                f -= -1
        pool = permutations([x for x in range(self.size)],self.size)
        pool = list(pool)
        allPermutations = []
        for i in range(len(pool)):
            allPermutations.append(list(pool[i]))
        
        collumnPermutations = individ.dislocate_permutations_verticaly()
        
        for i in range(len(collumnPermutations)):
            if(collumnPermutations[i] not in allPermutations):
                f += 1
        
                                
        return f
        
    
    
    def mutate(self,individ,chance):
        if(chance >= random.random()):
            i = random.randint(0,self.size-1)
            randomPerm = individ.combine_two_permutations(individ.get_size())
            try:
                individ.get_values()[i] = randomPerm
            except IndexError:
                print(i,self.size-1)
                sys.exit()
                
                
        
        return individ
        
    
    def crossover(self,parent1,parent2):
        child = []
        cut1 = random.randint(0,self.size-2)
        cut2 = random.randint(cut1,self.size-1)
        
        for i in range(cut1):
            child.append(parent1.get_values()[i])
            
        for i in range(cut1,cut2):
            child.append(parent2.get_values()[i])
            
        for i in range(cut2,self.size):
            child.append(parent1.get_values()[i])
            
        newIndivid = Individ(parent1.get_size())
        newIndivid.set_values(child)
        return newIndivid
        
    
    def swap_parent_child(self,parentIndex,child):
        self.problem.get_population()[parentIndex] = child
    
    def iteration_evolutive(self):
        #print(self.iterationNr)
        self.iterationNr += 1
        parent1Index = random.randint(0,self.count-1)
        parent2Index = random.randint(0,self.count-1)
       
        if (parent1Index != parent2Index):
            population = self.problem.get_population()
            parent1 = population[parent1Index]
            parent2 = population[parent2Index]
            child = self.crossover(parent1,parent2)
            child = self.mutate(child,self.chanceForMutation)
            
            
            fitnessParent1 = self.fitness(parent1)
            fitnessParent2 = self.fitness(parent2)
            childFitness = self.fitness(child)
            if(childFitness == 0):
                print("aici")
            if(fitnessParent1 < fitnessParent2 and childFitness < fitnessParent1):
                self.swap_parent_child(parent1Index,child)
            if(fitnessParent2 < fitnessParent1 and childFitness < fitnessParent2):
                self.swap_parent_child(parent2Index,child)
            return childFitness
        return 10
        
    
    def return_best_untill_now(self):
        graded = [ (self.fitness(x), x) for x in self.problem.get_population()]
        graded =  sorted(graded,key=lambda x: x[0])
        result=graded[0]
        fitnessOptim=result[0]
        individualOptim=result[1]
        print("best fitness : ",fitnessOptim)
        individualOptim.print_matrix()
            
    def solve(self):
        
        ok = 3
        while(ok != 0):
            ok = self.iteration_evolutive()
            if(ok < 5):
                self.return_best_untill_now()
            
        graded = [ (self.fitness(x), x) for x in self.problem.get_population()]
        graded =  sorted(graded,key=lambda x: x[0])
        
        
        result=graded[0]
        fitnessOptim=result[0]
        individualOptim=result[1]
        
        return fitnessOptim,individualOptim
        
        
class ProblemHillClimbing:
    
    def __init__(self,iterations,size):
        self.bestEver = Individ(size)
        self.bestEverFitness = 100
        self.iterations = iterations
        self.size = size
        self.individ = Individ(size)
        self.fitnessIndivid  = 100
        
    def change_best_ever(self,individ,fitness):
        
        self.bestEver = individ
        self.bestEverFitness = fitness
        print(self.bestEverFitness)
        self.bestEver.print_matrix()
        print(" ")
        print(" ")
        print(" ")
        
    def fitness(self,individ):
        f = 0
        matrix = individ.get_values()
        for i in range (len(matrix)):
            for j in range(len(matrix)):
                for k in range(len(matrix)):
                    for l in range(len(matrix)):
                        if(i != k or j != l):
                            if(matrix[i][j] == matrix[k][l]):
                                f -= -1
        pool = permutations([x for x in range(self.size)],self.size)
        pool = list(pool)
        allPermutations = []
        for i in range(len(pool)):
            allPermutations.append(list(pool[i]))
        
        collumnPermutations = individ.dislocate_permutations_verticaly()
        
        for i in range(len(collumnPermutations)):
            if(collumnPermutations[i] not in allPermutations):
                f += 1
        
                                
        return f
    
    def iteration_hill_climbing(self):
        neighbours = self.get_neighbours()
        self.fitnessIndivid = self.fitness(self.individ)
        found = False
        for neighbour in neighbours:
            neighbourFitness = self.fitness(neighbour)
            if(neighbourFitness < self.fitnessIndivid):
                self.individ = neighbour
                self.fitnessIndivid = neighbourFitness
                if(self.fitnessIndivid < self.bestEverFitness):
                    self.change_best_ever(neighbour, neighbourFitness)
                found = True
        if(found == False):
#            print("change")
           # self.individ.print_matrix()
           # print(self.fitnessIndivid)
            self.individ = Individ(self.size)
            
    def get_neighbours(self):
        permutationsInMatrix = self.individ.dislocate_permutations_horizontaly()
        neighbours = []
        for i in range(len(permutationsInMatrix)):
            permutationsNeighbours = permutations(permutationsInMatrix[i],self.size)
            permutationsNeighbours = list(permutationsNeighbours)
            for j in range(len(permutationsNeighbours)):
                newPermutations = copy.deepcopy(permutationsInMatrix)
                newPermutations[i] = permutationsNeighbours[j]
                
                newIndivid = Individ(self.size)
                newIndivid.set_matrix(newPermutations)
                neighbours.append(newIndivid)
        return neighbours
                
    
    
    def return_best_untill_now(self):
        return self.bestEverFitness,self.bestEver

    def solve(self):
        
        for i in range(self.iterations):
          #  print(self.fitnessIndivid)
            if(self.fitnessIndivid == 0):
                return self.fitnessIndivid,self.individ
            self.iteration_hill_climbing()
            
                
                
                    
                    
                    
            
            
        
        
        
        
        
        
class Controler:
    
    def __init__(self):
        self.problem = 5
        self.start = False
        
    def set_problem(self,problem):
        self.problem = problem
    
    def read_from_file(self):
        file = open("input.txt")
        popSize = int(file.readline())
        problemSize = int(file.readline())
        chanceForMutation = float(file.readline())
        iterations = int(file.readline())
        #return Problem(popSize,problemSize,chanceForMutation,iterations)
        return ProblemHillClimbing(int(iterations),int(problemSize))
    
    def solve(self):
        return self.problem.solve()
    
    def get_best(self):
        return self.problem.return_best_untill_now()
    
class Particle:
    
    
    def __init__(self,size):
        self.size = size
        self.values = []
        for i in range(size):
            self.values.append(self.combine_two_permutations(self.size))
        
        self.velocity = [ [[0,0] for x in range(self.size)] for x in range(self.size)]
        self.evaluate()
        self.bestFitness = self.fitness
        self.bestPosition = self.values
        
        
    def get_local_neighbours(self,nSize):
        neighbours = []
        allNeighbours = self.get_neighbours()
        for i in range(nSize):
            index = random.randint(0,len(allNeighbours)-1)
        
            neighbours.append(allNeighbours[index])
            del allNeighbours[index]
            
        return neighbours
    
    def get_neighbours(self):
        permutationsInMatrix = self.dislocate_permutations_horizontaly()
        neighbours = []
        for i in range(len(permutationsInMatrix)):
            permutationsNeighbours = permutations(permutationsInMatrix[i],self.size)
            permutationsNeighbours = list(permutationsNeighbours)
            for j in range(len(permutationsNeighbours)):
                newPermutations = copy.deepcopy(permutationsInMatrix)
                newPermutations[i] = permutationsNeighbours[j]
                
                newIndivid = Particle(self.size)
                newIndivid.set_matrix(newPermutations)
                neighbours.append(newIndivid)
        return neighbours
    
    
        
    def evaluate(self):
        self.fitness = self.fit()
        
    def get_velocity(self):
        return self.velocity
    
    def fit(self):
        f = 0
        matrix = self.get_values()
        for i in range (len(matrix)):
            for j in range(len(matrix)):
                for k in range(len(matrix)):
                    for l in range(len(matrix)):
                        if(i != k or j != l):
                            if(matrix[i][j] == matrix[k][l]):
                                f -= -1
        pool = permutations([x for x in range(self.size)],self.size)
        pool = list(pool)
        allPermutations = []
        for i in range(len(pool)):
            allPermutations.append(list(pool[i]))
        
        collumnPermutations = self.dislocate_permutations_verticaly()
        
        for i in range(len(collumnPermutations)):
            if(collumnPermutations[i] not in allPermutations):
                f += 1
        
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if(matrix[i][j][0] < 0 or matrix[i][j][0] >= self.size):
                    f += 2
                if(matrix[i][j][1] < 0 or matrix[i][j][1] >= self.size):
                    f += 2
                    
        return f
        
    def get_values(self):
        return self.values
    
    def __str__(self):
        return self.values
    def print_matrix(self):
        for i in range (self.size):
            print(self.values[i])
    
    def set_values(self,values):
        self.values = values
    
    
    def set_velocity(self,newVelocity):
        self.velocity = newVelocity
        
    def dislocate_one_permutation_horizontaly(self,i):
        permutation1 = []
        permutation2 = []
        for k in range(self.size):
            permutation1.append(self.values[i][k][0])
            permutation2.append(self.values[i][k][1])
        return permutation1,permutation2
    
    def dislocate_permutations_horizontaly(self):
        permutations = []
        for i in range(self.size):
            perm1,perm2 = self.dislocate_one_permutation_horizontaly(i)
            permutations.append(perm1)
            permutations.append(perm2)
        return permutations
    
    def discolate_one_permutation_verticaly(self,i):
        permutation1 = []
        permutation2 = []
        for k in range(self.size):
            permutation1.append(self.values[k][i][0])
            permutation2.append(self.values[k][i][1])
        return permutation1,permutation2
    
    def dislocate_permutations_verticaly(self):
        permutations = []
        for i in range(self.size):
            perm1,perm2 = self.discolate_one_permutation_verticaly(i)
            permutations.append(perm1)
            permutations.append(perm2)
        return permutations
    
    def get_random_permutaton(self,n):
        pool = permutations([x for x in range(n)],n)
        pool = list(pool)
        return list(pool[random.randint(0,len(pool)-1)])
        
        
    
    def combine_two_permutations(self,n):
        perm1 = self.get_random_permutaton(self.size)
        perm2 = self.get_random_permutaton(self.size)
        
        permutationForIndivid = []
        for i in range(len(perm1)):
            permutationForIndivid.append([perm1[i],perm2[i]])
        return permutationForIndivid
    
    def check_if_can_put_perm(self,permutationToAdd):
        for i in range(self.size):
            if(permutationToAdd == self.values[i]):
                return False
        return True
        
    def get_size(self):
        return self.size
    

    def set_matrix(self,permutations):
        matrix = []
        i = 0
        while(i < (len(permutations)-1)):
        
            row = []
            for j in range(self.size):
                cell = [permutations[i][j],permutations[i+1][j]]
                row.append(cell)
            matrix.append(row)
            i += 2
        self.set_values(matrix)
        
        
        
class ProblemPSO:
    
    def __init__(self,popSize,problemSize,w,c1,c2,nSize,iterations):
        self.iterations = iterations
        self.popSize = popSize
        self.problemSize = problemSize
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.population = [Particle(self.problemSize) for x in range(self.popSize)]
        self.nSize = nSize
        
      
    def print_best_particle(self):
        best = 0
        for i in range(self.popSize):
            if(self.population[i].fitness < self.population[best].fitness):
                best = i
        
        particle = self.population[best]
        
        print(particle.fitness)
        particle.print_matrix()
        
    def multiply_velocity_by_constant(self,v,constant):
        newVelocity = copy.deepcopy(v)
        for i in range(self.problemSize):
            for j in range(self.problemSize):
                newVelocity[i][j][0] *= constant
                newVelocity[i][j][1] *= constant
        return newVelocity
        
    def subtract_two_velocities(self,v1,v2):
        vel1 = copy.deepcopy(v1)
        vel2 = copy.deepcopy(v2)
        for i in range(self.problemSize):
            for j in range(self.problemSize):
                vel1[i][j][0] -= vel2[i][j][0]
                vel1[i][j][1] -= vel2[i][j][1]
                
        return vel1
                
    def add_two_velocities(self,v1,v2):
        vel1 = copy.deepcopy(v1)
        vel2 = copy.deepcopy(v2)
        for i in range(self.problemSize):
            for j in range(self.problemSize):
                vel1[i][j][0] += vel2[i][j][0]
                vel1[i][j][1] += vel2[i][j][1]
                
        return vel1
    
    def regulate_velocity(self,velocity):
        for i in range(self.problemSize):
            for j in range(self.problemSize):
                if(velocity[i][j][0] <= -self.problemSize):
                    velocity[i][j][0] = -self.problemSize
                if(velocity[i][j][0] >= self.problemSize):
                    velocity[i][j][0] = self.problemSize  
                if(velocity[i][j][1] <= -self.problemSize):
                    velocity[i][j][1] = -self.problemSize
                if(velocity[i][j][1] >= self.problemSize):
                    velocity[i][j][1] = self.problemSize 
    
    def iteration(self):
        bestNeighbours = []
        
        for i in range(self.popSize):
            bestNeighbour = Particle(self.problemSize)
            localNeighbours = self.population[i].get_local_neighbours(self.nSize)
            for particle in localNeighbours:
                particle.evaluate()
                if(particle.fitness < bestNeighbour.fitness):
                    bestNeighbour = particle
            bestNeighbours.append(bestNeighbour)
            
            
        for i in range(self.popSize):
            newVelocity = self.population[i].get_velocity()
            newVelocity = self.multiply_velocity_by_constant(newVelocity,self.w)
            bestNeighbourMinusCurrentVel = self.subtract_two_velocities(bestNeighbours[i].get_values(),self.population[i].get_values())
            bestNeighbourMinusCurrentVel = self.multiply_velocity_by_constant(bestNeighbourMinusCurrentVel, random.randint(0,self.problemSize-1))
            bestNeighbourMinusCurrentVel = self.multiply_velocity_by_constant(bestNeighbourMinusCurrentVel,self.c1)
            bestEverMinusCurrentVel = self.subtract_two_velocities(self.population[i].bestPosition,self.population[i].get_values())
            bestEverMinusCurrentVel = self.multiply_velocity_by_constant(bestNeighbourMinusCurrentVel,self.c2)
            bestEverMinusCurrentVel = self.multiply_velocity_by_constant(bestEverMinusCurrentVel, random.randint(0,self.problemSize-1))
            newVelocity = self.add_two_velocities(newVelocity,bestNeighbourMinusCurrentVel)
            newVelocity = self.add_two_velocities(newVelocity,bestEverMinusCurrentVel)
            self.regulate_velocity(newVelocity)
            self.population[i].set_velocity(newVelocity)
            
            
        
        newPopulation = copy.deepcopy(self.population)
        for i in range(self.popSize):
            particle = newPopulation[i]
            particle.set_values(self.add_two_velocities(newPopulation[i].get_velocity(),newPopulation[i].get_values()))
            particle.evaluate()
            if(particle.fitness < self.population[i].fitness):
                self.population[i] = particle
            if(particle.fitness < self.population[i].bestFitness):
                self.population[i].bestFitness = particle.fitness
                self.population[i].bestPosition = particle.get_values()
            
            
    def solve(self):
        for i in range(self.iterations):
            self.iteration()
            if(i % 20  == 0):
                self.print_best_particle()
        best = 0
        for i in range(self.popSize):
            if(self.population[i].fitness < self.population[best].fitness):
                best = i
        
        particle = self.population[best]
        
        print(particle.fitness)
        particle.print_matrix()
    
class show_progress(QWidget):
    
    def __init__(self,controler):
        super().__init__()
        self.controler = controler
        self.labels = []
        
        self.initUI()
    
    def initUI(self):
        vbox = QVBoxLayout()
        fitness,individ = self.controler.get_best()
        matrix = individ.get_values()
        for i in range (individ.size):
            offset = i * 20
            label = QLabel()
            label.setText(str(matrix[i]))
            self.labels.append(label)
            vbox.addWidget(label)
        label = QLabel()
        label.setText("fitness = " + str(fitness))
        self.labels.append(label)
        vbox.addWidget(label)
        
        
        self.setLayout(vbox)
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Best Solution found yet')    
        self.show()
            
        
    def keyPressEvent(self, e):
        
        if e.key() == Qt.Key_Space:
            fitness,individ = self.controler.get_best()
            matrix = individ.get_values()
            for i in range (individ.size):
                self.labels[i].setText(str(matrix[i]))
            self.labels[-1].setText(str(fitness))
           
            print("d ")
            
        






class SelectAlgorithmWindow(QWidget):
    
    def __init__(self,controler):
        super().__init__()
        self.controler = controler
        
        self.initUI()
        
        
    def initUI(self):      

        self.evolutiveButton = QPushButton('Select Evolutive', self)
        self.evolutiveButton.move(20, 20)
        self.evolutiveButton.clicked.connect(self.showDialogEvolutive)
        
        
        self.hillClimbingButton = QPushButton('Select Hill Climbing',self)
        self.hillClimbingButton.move(150,20)
        self.hillClimbingButton.clicked.connect(self.showDialogHill)
        
        self.hillClimbingButton = QPushButton('Select PSO',self)
        self.hillClimbingButton.move(300,20)
        self.hillClimbingButton.clicked.connect(self.showDialogPSO)
        
        
        self.solveButton = QPushButton('Solve', self)
        self.solveButton.move(70, 70)
        self.solveButton.clicked.connect(self.solve_the_problem)
        
        
        self.setGeometry(500, 500, 500, 500)
        self.setWindowTitle('Input dialog')
        self.show()
        
    
    def showDialogPSO(self):
        iterations, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the number of iterations:')
        problemSize, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the dimension of the matrix (ex :2 = 2x2, 3=3x3) enter 3 pls:')
        popSize, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the population size:')
        w, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the w(enter 1):')
        c1, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter c1 (enter 1):')
        c2, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter c2 (enter 2):')
        problemSize, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the dimension of the matrix (ex :2 = 2x2, 3=3x3):')
        nSize, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the neighbourhood size:')
        
        problem = ProblemPSO(int(popSize),int(problemSize),int(w),int(c1),int(c2),int(nSize),int(iterations))
        self.controler.set_problem(problem)
        
        
        
        
    def showDialogHill(self):
        iterations, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the number of iterations:')
        problemSize, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the dimension of the matrix (ex :2 = 2x2, 3=3x3):')
        
        problem = ProblemHillClimbing(int(iterations),int(problemSize))
        self.controler.set_problem(problem)
        
        
    def solve_the_problem(self):
        self.controler.start = True
        self.controler.solve()
        
        
        
    def showDialogEvolutive(self):
        
        iterations, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the number of iterations (ex: 100000):')
        problemSize, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the dimension of the matrix (ex :2 = 2x2, 3=3x3):')
        mutationChance, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the chance for mutation (ex 0.3 , 0.7):')
        popSize, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter the population size (ex :40):')
        
        problem = Problem(int(popSize), int(problemSize), float(mutationChance), int(iterations))
        self.controler.set_problem(problem)
        


def solve(controler):
    app = QApplication(sys.argv)
    ex = SelectAlgorithmWindow(controler)
    sys.exit(app.exec_())
    
    
def show(controler):
    while(controler.start == False):
        print("waiting")
        time.sleep(1)
    app = QApplication(sys.argv)
    
    ex = show_progress(controler)
    sys.exit(app.exec_())
                       
                       
                       
                       
                       
if __name__ == '__main__':
    controler = Controler()
    t = threading.Thread(target = solve,args=(controler,))
    t.start()
    t.join()    
    
    

def main():
    
    fitness,individ = controler.solve()
    print("fitness = ",fitness)
    individ.print_matrix()
#window()
#main()
              
        
        
        
                        
            
        











