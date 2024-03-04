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
        double[] weightedInputs = new double[numNodesOut];

        // Loop through outward connections
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double weightedInput = biases[nodeOut];

            // Loop through inward connections
            for (int nodeIn =  0; nodeIn < numNodesIn; nodeIn++)
            {
                weightedInput += inputs[nodeIn] * weights[nodeIn, nodeOut];
            }
        }

        return weightedInputs;
    }

}