namespace SimpleNN_Model;

public class Layer
{
    int numNodesIn, numNodesOut;
    double[,] weights;
    double[] biases;

    // Create the layer
    public Layer(int numNodesIn, int numNodesOut)
    {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;

        weights = new double[numNodesIn, numNodesOut];
        biases = new double[numNodesOut];
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
    double NodeCost(double outputActivation, double expectedOutput)
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