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
        NArray testOutput = NArray.FromElements(0f, 0f, 0f, 1f);

        NArray batchInput = NArray.CreateRandom(5, 3);
        NArray batchOutput = NArray.FromRows(
                new float[]{1f, 0f, 0f, 0f},
                new float[]{0f, 1f, 0f, 0f},
                new float[]{0f, 0f, 1f, 0f},
                new float[]{0f, 0f, 0f, 1f},
                new float[]{1f, 0f, 0f, 0f}
        );

        long start = System.currentTimeMillis();
        String[] startEnd = new String[2];


        // testing stochastic gradient descent methodology

        // single input loop
        for (int i = 0; i < 10000; i++) {
            network.input(testInput);
            network.forwardPass();
            network.calculateLoss(testOutput);
            if (i == 0) {
                startEnd[0] = "loss: " + network.loss() + " output: " + network.output.toString();
            }
            if (i == 9999) {
                startEnd[1] = "loss: " + network.loss() + " output: " + network.output.toString();
            }
            network.backwardPass(testOutput);
        }

        // batch loop
        for (int i = 0; i < 10000; i++) {
            network.input(batchInput);
            network.forwardPass();
            network.calculateLoss(batchOutput);
            if (i == 0) {
                startEnd[0] = "loss: " + network.loss() + " output: " + network.output.toString();
                network.printLayers();
            }
            if (i == 9999) {
                startEnd[1] = "loss: " + network.loss() + " output: " + network.output.toString();
                network.printLayers();
            }
            network.backwardPass(batchOutput);
        }

        long end = System.currentTimeMillis();
        System.out.println("time: " + (end - start) + " ms");
        System.out.println("first run: "+ startEnd[0] + "\nsecond run: " + startEnd[1]);

        //NeuralVisualizer.run(network);
    }
}
