package org.sdws;

import org.sdws.mathematics.*;
import org.sdws.nn.NetworkLayer;
import org.sdws.nn.NeuralNetwork;
import org.sdws.util.ActivationFunc;
import org.sdws.util.LossAlgorithm;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        NetworkLayer first = new NetworkLayer(3,6, ActivationFunc.ReLU);
        NetworkLayer second = new NetworkLayer(6,4, ActivationFunc.ReLU);
        NetworkLayer softmax = new NetworkLayer(4, 4, ActivationFunc.Softmax);
        NeuralNetwork network = new NeuralNetwork(
                first, second, softmax
        );

        NArray<Float> testInput = NArray.CreateRandom(3);
        System.out.println("testInput: " + testInput);

        NArray<Float> batchInput = NArray.CreateRandom(4,3);

        NArray<Float> testOutput = NArray.FromElements(0f, 0f, 0f, 1f);

        NArray<Float> batchOutput = NArray.FromRows(
                List.of(1f, 0f, 0f, 0f),
                List.of(1f, 0f, 0f, 0f),
                List.of(1f, 0f, 0f, 0f),
                List.of(1f, 0f, 0f, 0f)
        );

        network.feedForward(batchInput);
        network.calculateLoss(batchOutput, LossAlgorithm.CCELoss);
        System.out.println("loss: " + network.loss);
    }
}
