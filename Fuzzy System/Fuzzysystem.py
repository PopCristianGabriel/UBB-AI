# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:41:37 2020

@author: popcr
"""

from ast import literal_eval


class Functions:
    @staticmethod
    def trapezoidalRegion(a, b, c, d):
        return lambda x: max(0, min((x - a) / (b - a), 1, (d - x) / (d - c)))

    @staticmethod
    def triangularRegion(a, b, c):
        return Functions.trapezoidalRegion(a, b, b, c)

    @staticmethod
    def inverseLine(a, b):
        return lambda val: val * (b - a) + a

    @staticmethod
    def inverseTriangular(a, b, c):
        return lambda val: (Functions.inverseLine(a, b)(val) + Functions.inverseLine(c, b)(val)) / 2
    
    
class FuzzyRule:
    def __init__(self, inputVariables, outputVariables):
        self.outputVariablesVariables = outputVariables
        self.inputVariables = inputVariables

    def evaluateConjunction(self, inputVariables):
       
        result = min([inputVariables[description][outputName] for description, outputName in self.inputVariables.items()])
        print("conjunction = ", result)
        return [self.outputVariablesVariables, result]
    
    
class FuzzyDescriptions:
    def __init__(self):
        self.regions = {}
        self.Function = {}

    def add_region(self, region, membershipFunction, inverseMembershipFunction=None):
        self.regions[region] = membershipFunction
        self.Function[region] = inverseMembershipFunction

    def fuzzify(self, value):
       
        return {name: membershipFunction(value) for name, membershipFunction in self.regions.items()}

    def defuzzify(self, outputName, value):
        return self.Function[outputName](value)


class FuzzySystem:
    def __init__(self, rules):
        self.inDescriptions = {}
        self.outDescriptions = None
        self.rules = rules

    def add_description(self, name, outDescription, out=False):
        if out:
            if self.outDescriptions is None:
                self.outDescriptions = outDescription
        else:
            self.inDescriptions[name] = outDescription

    def evaluate(self, inputs):
        fuzzyInValues = self.evaluate_descriptions(inputs)
        rules = self.evaluate_fuzzy_rules(fuzzyInValues)

        fuzzyOutValues = [(list(description[0].values())[0], description[1]) for description in rules]
        weightedTotal = 0
        weightSum = 0
        for var in fuzzyOutValues:
            weightSum += var[1]
            print(var[1], "*", self.outDescriptions.defuzzify(*var) * var[1], "+")
            weightedTotal += self.outDescriptions.defuzzify(*var) * var[1]
        return weightedTotal / weightSum

    def evaluate_descriptions(self, inputs):
        return {name: self.inDescriptions[name].fuzzify(inputs[name]) for name, value in inputs.items()}

    def evaluate_fuzzy_rules(self, fuzzyInValues):
        return [rule.evaluateConjunction(fuzzyInValues) for rule in self.rules if rule.evaluateConjunction(fuzzyInValues)[1] != 0]


class Controller:
    def __init__(self, problemFile, inputFile, outputFile):
        self.out = outputFile
        self.loadData(problemFile, inputFile)

    def loadData(self, problemFile, inputFile):
        rules = []
        temperature = FuzzyDescriptions()
        humidity = FuzzyDescriptions()
        time = FuzzyDescriptions()
        file = open(problemFile,'r')
        
        data = file.read()
        data = data.splitlines()
        indexLine = 0
        indexLine = self.add_input_region(temperature, indexLine, data)
        indexLine = self.add_input_region(humidity, indexLine, data)
        indexLine = self.add_output_region(time, indexLine, data);
        self.add_rules(rules, indexLine, data)
        self.system = FuzzySystem(rules)
        self.system.add_description('temperature', temperature)
        self.system.add_description('humidity', humidity)
        self.system.add_description('time', time, out=True)
        self.read_problem_input(inputFile)

    def add_input_region(self, field, indexLine, splitInput):
        while splitInput[indexLine] != "----":
            line = splitInput[indexLine].split(",")
            if len(line) == 5:
                field.add_region(line[0], Functions.trapezoidalRegion(int(line[1]), 
                                int(line[2]), int(line[3]), int(line[4])))
            else:
                field.add_region(line[0], Functions.triangularRegion(int(line[1]),
                                int(line[2]), int(line[3])))
            indexLine += 1
        return indexLine + 1

    def add_rules(self, field, indexLine, inputData):
        while inputData[indexLine] != "----":
            line = inputData[indexLine].split(",")
            field.append(FuzzyRule({"temperature": line[0],'humidity': line[1]},
                                   {'time': line[2]}))
            indexLine += 1

    def add_output_region(self, field, i, inputData):
        while inputData[i] != "----":
            input1 = inputData[i].split(",")
            i += 1
            input2 = inputData[i].split(",")
            if len(input1) == 5:
                if len(input2) == 2:
                    field.add_region(input1[0], Functions.trapezoidalRegion(
                        int(input1[1]), int(input1[2]), int(input1[3]),
                        int(input1[4])), Functions.inverseLine(int(input2[0]),
                                                                  int(input2[1])))
                else:
                    field.add_region(input1[0], Functions.trapezoidalRegion(
                        int(input1[1]), int(input1[2]), int(input1[3]),
                        int(input1[4])), Functions.inverseTriangular(
                            int(input2[0]), int(input2[1]), 
                            int(input2[2])))
            else:
                if len(input2) == 2:
                    field.add_region(input1[0], Functions.triangularRegion(
                        int(input1[1]), int(input1[2]), int(input1[3])), 
                        Functions.inverseLine(int(input2[0]), int(input2[1])))
                else:
                    field.add_region(input1[0], Functions.triangularRegion(
                        int(input1[1]), int(input1[2]), int(input1[3])), 
                        Functions.inverseTriangular(int(input2[0]), 
                                                    int(input2[1]), 
                                                    int(input2[2])))
            i += 1
        return i + 1

    def read_problem_input(self, inputData):
        file = open(inputData,'r')
        data = file.read()
        data = data.splitlines();
        self.inputs = {data[0].split(",")[0]: int(data[0].split(",")[1]),
                       data[1].split(",")[0]: int(data[1].split(",")[1])}


    def get_inputs(self):
        return self.inputs

    def evaluate(self):
        result = str(self.system.evaluate(self.inputs))
        file = open(self.out,'w')
        file.write(result)
        return result
    
    
    
def main():
    controller = Controller("problem.in", "input.in", "output.out")
    result = "time: " + controller.evaluate() 
    print(result)
    
main()