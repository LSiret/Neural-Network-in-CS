namespace SimpleNN_Model;

public class NeuralNetwork
{

    Layer[] layers;

    // Create the Neural Network
    public NeuralNetwork(params int[] layerSizes)
    {
        layers = new Layer[layerSizes.Length - 1];

        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
        }
    }

    // Run input values through network to calculate output
    double[] CalculateOutputs(double[] inputs)
    {
        foreach(Layer layer in layers)
        {
            inputs = layer.CalculateOutputs(inputs);
        }

        return inputs;
    }

    // Calculate the error cost of a single data point
    double Cost(DataPoint dataPoint)
    {
        double[] outputs = CalculateOutputs(dataPoint.inputs);
        Layer outputLayer = layers[layers.Length - 1];
        double cost = 0;

        for (int nodeOut = 0; nodeOut < outputs.Length; nodeOut++)
        {
            cost += outputLayer.NodeCost(outputs[nodeOut], dataPoint.expectedOutputs[nodeOut]);
        }

        return cost;
    }

    // Calculate average error cost over all data points
    double Cost(DataPoint[] data)
    {
        double totalCost = 0;

        foreach(DataPoint dataPoint in data)
        {
            totalCost += Cost(dataPoint);
        }

        return totalCost / data.Length;
    }

    public void Learn(DataPoint[] trainingData, double learnRate)
    {
        const double h = 0.00001;
        double originalCost = Cost(trainingData);

        foreach (Layer layer in layers)
        {
            // Calculate the cost gradient for the current weight
            for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++)
            {
                for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++)
                {

                }
            }
        }
    }

    // Run the inputs through the network and calculate which node has the highest value
    int Classify(double[] inputs)
    {
        double[] outputs = CalculateOutputs(inputs);
        return IndexOfMaxValue(outputs);
    }

    // Get the index of the largest value in an array
    int IndexOfMaxValue(double[] inputs)
    {
        int maxIndex = 0;
        int maxValue = 0;

        for (int i = 0;i < inputs.Length;i++)
        {
            if (inputs[i] > maxValue)
            {
                maxIndex = i;
            }
        }

        return maxIndex;
    }

}