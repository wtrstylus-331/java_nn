package org.sdws;

import org.sdws.mathematics.*;
import org.sdws.nn.NetworkLayer;
import org.sdws.nn.NeuralNetwork;
import org.sdws.util.ActivationFunc;
import org.sdws.util.LossAlgorithm;
import org.sdws.visualizer.NeuralVisualizer;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        NetworkLayer first = new NetworkLayer(3,6, ActivationFunc.ReLU);
        NetworkLayer second = new NetworkLayer(6,4, ActivationFunc.ReLU);
        NetworkLayer softmax = new NetworkLayer(4, 4, ActivationFunc.Softmax);
        NeuralNetwork network = new NeuralNetwork(
                0.001f,
                LossAlgorithm.CCELoss,
                first, second, softmax
        );

        NArray testInput = NArray.CreateRandom(3);
        System.out.println("testInput: " + testInput);

        NArray testOutput = NArray.FromElements(0f, 0f, 0f, 1f);

        NArray batchOutput = NArray.FromRows(
                new float[]{1f, 0f, 0f, 0f},
                new float[]{0f, 1f, 0f, 0f},
                new float[]{0f, 0f, 1f, 0f},
                new float[]{0f, 0f, 0f, 1f}
        );

        long start = System.currentTimeMillis();

        network.input(testInput);
        network.forwardPass();
        network.calculateLoss(testOutput);
        System.out.println("loss: " + network.loss() + " output: " + network.output);
        network.backwardPass(testOutput);

        long end = System.currentTimeMillis();
        System.out.println("time: " + (end - start) + " ms");

        //NeuralVisualizer.run(network);
    }
}
