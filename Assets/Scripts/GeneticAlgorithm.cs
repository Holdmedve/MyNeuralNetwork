using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Matlab;
using System.IO;


public class GeneticAlgorithm : MonoBehaviour
{
    public BallController ballController;
    [Header("Settings")]
    public int populationSize = 10;
    [Range(0.0f, 1.0f)]
    public float mutationRate = 0.01f;
    [Range(0, 50)]
    public int numOfEliteChildren = 6;


    [Header("Cool numbers")]
    public int currentGeneration;
    public int currentGenome;
    public List<int> genePool;
    public NeuralNet[] population;
    public int topFitness;
    private List<int> eliteIndeces;

    // Start is called before the first frame update
    void Start()
    {
        genePool = new List<int>();
        eliteIndeces = new List<int>();
        topFitness = 0;
        currentGeneration = 0;
        currentGenome = 0;
        CreatePopulation();
    }

    void CreatePopulation()
    {
        population = new NeuralNet[populationSize];
        for (int i = 0; i < populationSize; i++)
        {
            population[i] = new NeuralNet();
            population[i].Initialize();
            population[i].RandomiseWeightsAndBiases();
        }
        ballController.ResetToCurrentNetwork(population[currentGenome]);
    }


    public void Death(NeuralNet nn, float timeLimit, float timeSinceStart)
    {
        if (nn.fitness > topFitness && timeSinceStart < timeLimit)
        {
            topFitness = (int)nn.fitness;
            SaveNeuralNet(population[currentGenome]);
        }

        population[currentGenome].fitness = nn.fitness;
        population[currentGenome].scoredGoal = nn.scoredGoal;
        Debug.Log("currentGenome:\t" + currentGenome + "\tfitness:\t" + (int)nn.fitness + "\tscoredGoal:\t" + nn.scoredGoal);
        currentGenome++;

        if (currentGenome < populationSize)
        {
            ballController.ResetToCurrentNetwork(population[currentGenome]);
        }
        else
        {
            WriteToTxt();
            if (currentGeneration == 60)
            {
                UnityEditor.EditorApplication.isPlaying = false;
            }
            else
            {
                Repopulate();
            }
        }
    }

    void Repopulate()
    {
        genePool = new List<int>();
        eliteIndeces = new List<int>();
        currentGenome = 0;
        currentGeneration++;

        // initialize new population
        NeuralNet[] newPopulation = new NeuralNet[populationSize];
        for (int i = 0; i < populationSize; i++)
        {
            newPopulation[i] = new NeuralNet();
            newPopulation[i].Initialize();
        }

        SelectEliteChildren(newPopulation);
        CarryEliteChildrenOver(ref newPopulation);
        FillTheGenePool();
        Crossover(ref newPopulation);
        Mutate(ref newPopulation);


        population = new NeuralNet[populationSize]; // do you need this line?:/... leave it
        for (int i = 0; i < populationSize; i++)
        {
            population[i] = newPopulation[i];
        }

        //population = newPopulation;
        ballController.ResetToCurrentNetwork(population[currentGenome]);

    }

    void SelectEliteChildren(NeuralNet[] newPopulation)
    {
        List<KeyValuePair<int, float>> fitnessValues = new List<KeyValuePair<int, float>>();
        for (int i = 0; i < populationSize; i++)
        {
            fitnessValues.Add(new KeyValuePair<int, float>(i, population[i].fitness));
        }

        // sort the list
        for (int i = 0; i < populationSize - 1; i++)
        {
            int maxIdx = i;
            for (int j = i + 1; j < populationSize; j++)
            {
                if (fitnessValues[j].Value > fitnessValues[maxIdx].Value)
                {
                    maxIdx = j;
                }

            }
            if (maxIdx != i)
            {
                KeyValuePair<int, float> temp = new KeyValuePair<int, float>();
                temp = fitnessValues[i];

                fitnessValues[i] = new KeyValuePair<int, float>();
                fitnessValues[i] = fitnessValues[maxIdx];

                fitnessValues[maxIdx] = temp;
            }
        }

        // DEBUG
        string s = "";
        for (int i = 0; i < populationSize; i++)
        {
            s += "idx:\t" + fitnessValues[i].Key + "\tfitness:\t" + fitnessValues[i].Value + "\n";
        }
        Debug.Log(s);


        for (int i = 0; i < numOfEliteChildren; i++)
        {
            eliteIndeces.Add(fitnessValues[i].Key);
        }
    }

    void CarryEliteChildrenOver(ref NeuralNet[] newPopulation)
    {
        for (int i = 0; i < numOfEliteChildren; i++)
        {
            newPopulation[i] = population[eliteIndeces[i]];
        }
    }

    void FillTheGenePool()
    {
        for (int i = 0; i < populationSize; i++)
        {
            int occurenceInGenePool = (int)(population[i].fitness);
            for (int j = 0; j < occurenceInGenePool; j++)
            {
                genePool.Add(i);
            }
        }

        string s = "";
        for (int i = 0; i < genePool.Count; i++)
        {
            s += genePool[i] + "\t";
        }
        Debug.Log(s);


    }

    void Crossover(ref NeuralNet[] newPopulation) // REF... MAYBE DELETE IT
    {
        int numOfWeights = newPopulation[0].hiddenNeuronCount * 4; // this is only true with 2 input and 2 output neurons
        int numOfBiases = newPopulation[0].hiddenNeuronCount + newPopulation[0].outputNeuronCount;

        // (0, 1, ..., numOfEliteChildren - 1) those are elite, therefore they mustn't be overwritten with crossover
        for (int i = numOfEliteChildren; i < populationSize; i += 2)
        {
            int A_parentIdx = -1;
            int B_parentIdx = -1;

            for (int j = 0; j < 10; j++)
            {
                A_parentIdx = genePool[Random.Range(0, genePool.Count)];
                B_parentIdx = genePool[Random.Range(0, genePool.Count)];
                if (A_parentIdx != B_parentIdx)
                {
                    break;
                }
            }

            int weightCrossOverPoint = Random.Range(1, numOfWeights - 1);
            int count = 0;
            for (int j = 0; j < 2; j++)
            {
                for (int col = 0; col < newPopulation[i].weights[j].ColumnCount; col++)
                {
                    for (int row = 0; row < newPopulation[i].weights[j].RowCount; row++)
                    {
                        if (count < weightCrossOverPoint)
                        {
                            newPopulation[i].weights[j][row, col] = population[A_parentIdx].weights[j][row, col];
                            newPopulation[i + 1].weights[j][row, col] = population[B_parentIdx].weights[j][row, col];
                        }
                        else
                        {
                            newPopulation[i].weights[j][row, col] = population[B_parentIdx].weights[j][row, col];
                            newPopulation[i + 1].weights[j][row, col] = population[A_parentIdx].weights[j][row, col];
                        }
                        count++;
                    }
                }
            }

            int biasCrossOverPoint = Random.Range(1, numOfBiases - 1);
            count = 0;
            for (int j = 0; j < 2; j++)
            {
                for (int col = 0; col < newPopulation[i].biases[j].ColumnCount; col++)
                {
                    if (count < biasCrossOverPoint)
                    {
                        newPopulation[i].biases[j][0, col] = population[A_parentIdx].biases[j][0, col];
                        newPopulation[i + 1].biases[j][0, col] = population[B_parentIdx].biases[j][0, col];
                    }
                    else
                    {
                        newPopulation[i].biases[j][0, col] = population[B_parentIdx].biases[j][0, col];
                        newPopulation[i + 1].biases[j][0, col] = population[A_parentIdx].biases[j][0, col];
                    }
                }
            }
        }
    }

    void Mutate(ref NeuralNet[] newPopulation)
    {
        for (int i = numOfEliteChildren; i < populationSize; i++)
        {
            if (Random.Range(0.0f, 1.0f) < mutationRate) // mutation means one weight and one bias is randomized
            {
                // weight
                int weightMatIdx = Random.Range(0, newPopulation[i].weights.Count);
                int weightRowIdx = Random.Range(0, newPopulation[i].weights[weightMatIdx].RowCount);
                int weightColIdx = Random.Range(0, newPopulation[i].weights[weightMatIdx].ColumnCount);

                newPopulation[i].weights[weightMatIdx][weightRowIdx, weightColIdx] = Random.Range(0.0f, 1.0f);

                // bias
                int biasMatIdx = Random.Range(0, newPopulation[i].biases.Count);
                int biasRowIdx = Random.Range(0, newPopulation[i].biases[biasMatIdx].RowCount);
                int biasColIdx = Random.Range(0, newPopulation[i].biases[biasMatIdx].ColumnCount);

                newPopulation[i].biases[biasMatIdx][biasRowIdx, biasColIdx] = Random.Range(0.0f, 1.0f);
            }
        }
    }


    void SaveNeuralNet(NeuralNet nn)
    {
        var matrices = new List<MatlabMatrix>();

        matrices.Add(MatlabWriter.Pack(nn.hiddenLayer, name));

        // adding biases
        for (int i = 0; i < nn.biases.Count; i++)
        {
            string name = "bias" + i.ToString();
            matrices.Add(MatlabWriter.Pack(nn.biases[i], name));
        }

        // adding weights
        for (int i = 0; i < nn.weights.Count; i++)
        {
            string name = "weight" + i.ToString();
            matrices.Add(MatlabWriter.Pack(nn.weights[i], name));
        }

        System.DateTime now = System.DateTime.Now;

        string path = "../ShowcaseNeuralNet/selectedNNs/fitness_" + nn.fitness.ToString() + ".mat";
        MathNet.Numerics.Data.Matlab.MatlabWriter.Store(path, matrices);
    }

    public void WriteToTxt()
    {
        int neuronCount = population[0].hiddenNeuronCount;
        string path = "Evaluation/popSize_" + populationSize + "_mutRate_" + mutationRate + "_neuCount_" + neuronCount + ".txt";

        if (!File.Exists(path))
        {
            // create file
            using (StreamWriter sw = File.CreateText(path))
            {

            }
        }

        using (StreamWriter sw = File.AppendText(path))
        {
            int meanFitness = 0;
            for (int i = 0; i < populationSize; i++)
            {
                meanFitness += (int)population[i].fitness;
            }
            meanFitness /= populationSize;

            int goalCount = 0;
            for (int i = 0; i < populationSize; i++)
            {
                if (population[i].scoredGoal)
                {
                    goalCount++;
                }
            }

            string s = currentGeneration + "\t" + meanFitness.ToString() + "\t" + goalCount;

            sw.WriteLine(s);
        }
    }
}

