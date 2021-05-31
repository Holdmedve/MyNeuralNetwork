import matplotlib.pyplot as plt
import numpy as np
import glob
import os


path1 = 'Evaluation/popSize_30_mutRate_0.01_neuCount_3.txt'
path2 = 'Evaluation/popSize_30_mutRate_0.01_neuCount_6.txt'
path3 = 'Evaluation/popSize_30_mutRate_0.05_neuCount_3.txt'
path4 = 'Evaluation/popSize_30_mutRate_0.05_neuCount_6.txt'

path5 = 'Evaluation/popSize_80_mutRate_0.01_neuCount_3.txt'
path6 = 'Evaluation/popSize_80_mutRate_0.01_neuCount_6.txt'
path7 = 'Evaluation/popSize_80_mutRate_0.05_neuCount_3.txt'
path8 = 'Evaluation/popSize_80_mutRate_0.05_neuCount_6.txt'

paths = list()
asd = 0

if asd == 1:
    paths.append(path1)
    paths.append(path2)
    paths.append(path3)
    paths.append(path4)
else:
    paths.append(path5)
    paths.append(path6)
    paths.append(path7)
    paths.append(path8)


plt.rcParams.update({'font.size': 15, 'font.weight': 'bold'})
palette = plt.get_cmap('Set1')
plt.style.use('seaborn-darkgrid')

fittingDegree = 9

for path in paths:
    with open(path) as f:
        lines = f.readlines()
        generation = [float(line.split('\t')[0]) for line in lines]
        meanFitness = [float(line.split('\t')[1]) for line in lines]

        coefficients = np.polyfit(generation, meanFitness, fittingDegree)
        poly = np.poly1d(coefficients)

        new_x = np.linspace(generation[0], generation[-1])
        new_y = poly(new_x)

        plt.xlabel('generation')
        plt.ylabel('mean fitness')

        plt.plot(new_x, new_y, marker='.', linewidth=1)

if asd == 1:
    plt.title('population size = 30')
    plt.legend(['mutRate: 0.01 neuCount: 3',
                'mutRate: 0.01 neuCount: 6',
                'mutRate: 0.05 neuCount: 3',
                'mutRate: 0.05 neuCount: 6'])
else:
    plt.title('population size = 80')
    plt.legend(['mutRate: 0.01 neuCount: 3',
                'mutRate: 0.01 neuCount: 6',
                'mutRate: 0.05 neuCount: 3',
                'mutRate: 0.05 neuCount: 6'])

plt.show()

for path in paths:
    with open(path) as f:
        lines = f.readlines()
        generation = [float(line.split('\t')[0]) for line in lines]
        numOfGoals = [float(line.split('\t')[2]) for line in lines]

        coefficients = np.polyfit(generation, numOfGoals, fittingDegree)
        poly = np.poly1d(coefficients)

        new_x = np.linspace(generation[0], generation[-1])
        new_y = poly(new_x)

        plt.xlabel('generation')
        plt.ylabel('number of goals')
        plt.plot(new_x, new_y, marker='.', linewidth=1)


if asd == 1:
    plt.title('population size = 30')
    plt.legend(['mutRate: 0.01 neuCount: 3',
                'mutRate: 0.01 neuCount: 6',
                'mutRate: 0.05 neuCount: 3',
                'mutRate: 0.05 neuCount: 6'])
else:
    plt.title('population size = 80')
    plt.legend(['mutRate: 0.01 neuCount: 3',
                'mutRate: 0.01 neuCount: 6',
                'mutRate: 0.05 neuCount: 3',
                'mutRate: 0.05 neuCount: 6'])
plt.show()
