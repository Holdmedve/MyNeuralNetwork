using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MathNet.Numerics.LinearAlgebra;
using Random = UnityEngine.Random;


public class NeuralNet : MonoBehaviour
{
    public int inputNeuronCount = 2;
    public int hiddenNeuronCount = 6;
    public int outputNeuronCount = 2;

    public Matrix<float> inputLayer;
    public Matrix<float> hiddenLayer;
    public Matrix<float> outputLayer;
    public List<Matrix<float>> weights;
    public List<Matrix<float>> biases;

    public float fitness;
    public bool scoredGoal;


    public void Initialize()
    {
        fitness = 0f;
        scoredGoal = false;

        // layers
        inputLayer = Matrix<float>.Build.Dense(1, inputNeuronCount);
        hiddenLayer = Matrix<float>.Build.Dense(1, hiddenNeuronCount);
        outputLayer = Matrix<float>.Build.Dense(1, outputNeuronCount);

        // weights
        weights = new List<Matrix<float>>();
        weights.Add(Matrix<float>.Build.Dense(inputNeuronCount, hiddenNeuronCount));
        weights.Add(Matrix<float>.Build.Dense(hiddenNeuronCount, outputNeuronCount));

        // biases
        biases = new List<Matrix<float>>();
        biases.Add(Matrix<float>.Build.Dense(1, hiddenNeuronCount));
        biases.Add(Matrix<float>.Build.Dense(1, outputNeuronCount));
    }

    public void RandomiseWeightsAndBiases()
    {
        // weights
        for (int i = 0; i < weights.Count; i++)
        {
            for (int x = 0; x < weights[i].RowCount; x++)
            {
                for (int y = 0; y < weights[i].ColumnCount; y++)
                {
                    weights[i][x, y] = Random.Range(-1f, 1f);
                }
            }
        }

        // biases
        for (int i = 0; i < biases.Count; i++)
        {
            for (int j = 0; j < biases[i].ColumnCount; j++)
            {
                biases[i][0, j] = Random.Range(-1f, 1f);
            }
        }
    }

    public (float, float) RunNetwork(float[] inputs)
    {
        for (int i = 0; i < inputs.Length; i++)
        {
            inputLayer[0, i] = inputs[i]; // loading the 2 distance to box values
        }

        inputLayer.PointwiseTanh();
        hiddenLayer = (inputLayer * weights[0] + biases[0]).PointwiseTanh();


        outputLayer = hiddenLayer * weights[1] + biases[1];
        outputLayer = outputLayer.PointwiseTanh();

        // MoveDirectionX, MoveDirectionZ
        return (outputLayer[0, 0], outputLayer[0, 1]);
    }
}
