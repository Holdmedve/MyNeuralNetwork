using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallController : MonoBehaviour
{

    private Rigidbody ballRigidBody;
    public Rigidbody boxRigidBody;
    private NeuralNet neuralNet;
    private float[] inputs;

    [Header("Inputs")]

    public float directionToBoxX;
    public float directionToBoxZ;

    [Header("outputs")]
    public float moveDirectionX;
    public float mvoeDirectionZ;

    [Header("Fitness related data")]
    public float fitness;
    public float timeSinceStart;
    public float boxVelocityX;

    [Header("Box data")]

    public float boxBallDistance;

    [Header("Settings")]
    public float timeLimit = 10f;
    public float boxVelocityXMultiplier = 0.01f;

    [Header("Cool numbers")]
    public Vector3 directionToBox;
    public Vector3 ballVelocity;
    public float ballVelMagnitude;
    public float angle;

    // ball
    private Vector3 ballStartPosition;
    // box
    private Vector3 boxStartPosition;

    // Start is called before the first frame update
    void Awake()
    {
        inputs = new float[2];
        ballRigidBody = GetComponent<Rigidbody>();
        neuralNet = GetComponent<NeuralNet>();

        SpawnBallandBox();



        fitness = 0f;
    }

    void SpawnBallandBox()
    {
        float X_upperBoundary = 15f;
        float X_lowerBoundary = 5f;
        float Z_upperBoundary = 11f;
        float Z_lowerBoundary = 4f;

        float minSpawnDistance = 2f; // between the box and the ball

        do
        {
            boxStartPosition = new Vector3(Random.Range(X_lowerBoundary, X_upperBoundary), 1, Random.Range(Z_lowerBoundary, Z_upperBoundary));
            ballStartPosition = new Vector3(Random.Range(X_lowerBoundary, X_upperBoundary), 1, Random.Range(Z_lowerBoundary, Z_upperBoundary));
        } while (Vector3.Distance(boxStartPosition, ballStartPosition) < minSpawnDistance);

        ballRigidBody.position = ballStartPosition;
        ballRigidBody.rotation = Quaternion.identity;
        ballRigidBody.velocity = Vector3.zero;
        ballRigidBody.angularVelocity = Vector3.zero;

        boxRigidBody.position = boxStartPosition;
        boxRigidBody.rotation = Quaternion.identity;
        boxRigidBody.velocity = Vector3.zero;
        ballRigidBody.angularVelocity = Vector3.zero;
    }

    public void ResetToCurrentNetwork(NeuralNet nn)
    {
        this.neuralNet = nn;
        timeSinceStart = 0f;
        fitness = 0f;

        SpawnBallandBox();
    }


    private void FixedUpdate()
    {
        GatherInputs();
        (moveDirectionX, mvoeDirectionZ) = neuralNet.RunNetwork(inputs);
        MoveBall(moveDirectionX, mvoeDirectionZ);

        timeSinceStart += Time.deltaTime;
        boxVelocityX = boxRigidBody.velocity.x;

        directionToBox = (boxRigidBody.position - ballRigidBody.position);
        angle = Vector3.Angle(ballRigidBody.velocity, directionToBox);

        ballVelocity = ballRigidBody.velocity;
        ballVelMagnitude = ballVelocity.magnitude;

        CalculateFitness();
    }



    void CalculateFitness()
    {
        // increase fitness
        if (ballVelMagnitude > 0.1f && angle < 80f) // ball moves towards box
        {
            fitness += ballVelMagnitude * 0.0001f * (80 - angle);
        }
        if (fitness + boxVelocityX * boxVelocityXMultiplier >= 0)
        {
            fitness += boxVelocityX * boxVelocityXMultiplier;
        }


        // kill the ball
        if (timeSinceStart > timeLimit)
        {
            neuralNet.fitness = fitness;
            GameObject.FindObjectOfType<GeneticAlgorithm>().Death(neuralNet, fitness, timeSinceStart);
        }
        if (boxRigidBody.position.x >= 20f) // finish line
        {
            fitness += (int)(timeLimit - timeSinceStart) * 10;
            neuralNet.fitness = fitness; // DO U NEED TO SET THE FITNESS VALUE IN THE DEATH FUNCTION AS WELL?
            neuralNet.scoredGoal = true;
            GameObject.FindObjectOfType<GeneticAlgorithm>().Death(neuralNet, fitness, timeSinceStart);
        }
    }

    private float speed = 20;
    void MoveBall(float directionX, float directionZ)
    {
        Vector3 forceDirection = Vector3.zero;
        forceDirection.x = directionX;
        forceDirection.z = directionZ;

        ballRigidBody.AddForce(forceDirection * speed);
    }

    void GatherInputs()
    {
        directionToBox = (boxRigidBody.position - ballRigidBody.position);
        directionToBoxX = directionToBox.x;
        directionToBoxZ = directionToBox.z;

        inputs[0] = directionToBoxX;
        inputs[1] = directionToBoxZ;
    }
}
