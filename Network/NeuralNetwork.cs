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