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
                0.0001f,
                LossAlgorithm.CCELoss,
                first, second, softmax
        );

        NetworkLayer first2 = new NetworkLayer(3,6, ActivationFunc.ReLU);
        NetworkLayer second2 = new NetworkLayer(6,4, ActivationFunc.ReLU);
        NetworkLayer softmax2 = new NetworkLayer(4, 4, ActivationFunc.Softmax);

        NArray wF2 = NArray.DeepCopy(first.getWeights());
        NArray wS2 = NArray.DeepCopy(second.getWeights());
        NArray wSx2 = NArray.DeepCopy(softmax.getWeights());

        first2.setWeights(wF2);
        second2.setWeights(wS2);
        softmax2.setWeights(wSx2);

        NeuralNetwork network2 = new NeuralNetwork(
                0.0001f,
                LossAlgorithm.CCELoss,
                first2, second2, softmax2
        );

        NArray data = NArray.FromRows(
                new float[]{-0.8228878f, 0.17324293f, -0.06841123f},
                new float[]{-0.9486228f, -0.84785914f, -0.84727573f},
                new float[]{0.633168f, -0.75513864f, 0.1217463f},
                new float[]{0.2956028f, 0.23604989f, 0.9242351f}
        );
        NArray batchOutput = NArray.FromRows(
                new float[]{1f, 0f, 0f, 0f},
                new float[]{0f, 1f, 0f, 0f},
                new float[]{0f, 0f, 1f, 0f},
                new float[]{0f, 0f, 0f, 1f}
        );

        System.out.println("\nnetwork before:");
        for (int i = 0; i < data.shape[0]; i++) {
            network.input(NArray.FromElements(data.getMatrixArray()[i]));
            network.forwardPass();
            System.out.println(network.output);
        }
        System.out.println();

        sgd(network, data, batchOutput, 50000);

        System.out.println("\nnetwork after:");
        for (int i = 0; i < data.shape[0]; i++) {
            network.input(NArray.FromElements(data.getMatrixArray()[i]));
            network.forwardPass();
            System.out.println(network.output);
        }
        System.out.println("\n=====================================================\n");

        System.out.println("\nnetwork before:");
        for (int i = 0; i < data.shape[0]; i++) {
            network2.input(NArray.FromElements(data.getMatrixArray()[i]));
            network2.forwardPass();
            System.out.println(network2.output);
        }
        System.out.println();

        fullBatch(network2, data, batchOutput, 50000);

        System.out.println("\nnetwork after:");
        System.out.println();
        for (int i = 0; i < data.shape[0]; i++) {
            network2.input(NArray.FromElements(data.getMatrixArray()[i]));
            network2.forwardPass();
            System.out.println(network2.output);
        }

        //NeuralVisualizer.run(network);
    }

    static void sgd(NeuralNetwork network, NArray dataset, NArray outputset, int iterations) {
        System.out.println("SGD:");
        float[][] desiredArray = outputset.getMatrixArray();
        float[][] datasetArray = dataset.getMatrixArray();

        long start = System.currentTimeMillis();
        String[] startEnd = new String[2];

        for (int i = 0; i < iterations; i++) {
            for (int j = 0; j < datasetArray.length; j++) {
                NArray vec = NArray.FromElements(datasetArray[j]);
                NArray desiredVec = NArray.FromElements(desiredArray[j]);

                network.input(vec);
                network.forwardPass();
                network.calculateLoss(desiredVec);
                if (i == 0) {
                    startEnd[0] = "loss: " + network.loss() + " output: " + network.output.toString();
                }
                network.backwardPass(desiredVec);
            }
        }
        startEnd[1] = "loss: " + network.loss() + " output: " + network.output.toString();
        long end = System.currentTimeMillis();
        System.out.println("time: " + (end - start) + " ms");
        System.out.println("first run: "+ startEnd[0] + "\nlast run: " + startEnd[1]);
    }

    static void fullBatch(NeuralNetwork network, NArray dataset, NArray outputset, int iterations) {
        System.out.println("Full Batch GD:");

        long start = System.currentTimeMillis();
        String[] startEnd = new String[2];

        for (int i = 0; i < iterations; i++) {
            network.input(dataset);
            network.forwardPass();
            network.calculateLoss(outputset);
            if (i == 0) {
                startEnd[0] = "loss: " + network.loss() + " output: " + network.output.toString();
            }
            network.backwardPass(outputset);
        }
        startEnd[1] = "loss: " + network.loss() + " output: " + network.output.toString();

        long end = System.currentTimeMillis();
        System.out.println("time: " + (end - start) + " ms");
        System.out.println("first run: "+ startEnd[0] + "\nlast run: " + startEnd[1]);
    }
}
