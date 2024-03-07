namespace SimpleNN_Model;

public class Layer
{
    public int numNodesIn, numNodesOut;
    public double[,] costGradientW;
    public double[] costGradientB;
    public double[,] weights;
    public double[] biases;

    // Create the layer
    public Layer(int numNodesIn, int numNodesOut)
    {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;

        costGradientW = new double[numNodesIn, numNodesOut];
        weights = new double[numNodesIn, numNodesOut];
        costGradientB = new double[numNodesOut];
        biases = new double[numNodesOut];
        
        InitialiseRandomWeights();
        
    }

    // Generate random weights for all nodes
    public void InitialiseRandomWeights()
    {
        Random rng = new Random();

        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
        {
            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                // Generate random number between -1 and 1
                double randomValue = rng.NextDouble() * 2 - 1;

                // Scale random value to 1 / sqrt num of inputs
                weights[nodeIn, nodeOut] = randomValue / Math.Sqrt(numNodesIn);

            }
        }
    }

    // Update weights and biases based on cost gradients
    public void ApplyGradients(double learnRate)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            biases[nodeOut] -= costGradientB[nodeOut] * learnRate;
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weights[nodeIn, nodeOut] -= costGradientW[nodeIn, nodeOut] * learnRate;
            }
        }
    }

    // Calculate the output of the layer
    public double[] CalculateOutputs(double[] inputs)
    {
        double[] activationValues = new double[numNodesOut];

        // Loop through outward connections
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double weightedInput = biases[nodeOut];

            // Loop through inward connections
            for (int nodeIn =  0; nodeIn < numNodesIn; nodeIn++)
            {
                // Calculate value of the output node from its inputs
                weightedInput += inputs[nodeIn] * weights[nodeIn, nodeOut];
            }

            // Set outward node to calculated value
            activationValues[nodeOut] = ActivationFunction(weightedInput);
        }

        return activationValues;
    }

    // Get the square of the error in a layer
    public double NodeCost(double outputActivation, double expectedOutput)
    {
        double error = outputActivation - expectedOutput;
        return error * error;
    }

    double ActivationFunction(double input)
    {
        // ReLU function
        return Math.Max(0, input);
    }

}