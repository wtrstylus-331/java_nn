package org.sdws;

import org.sdws.mathematics.*;
import org.sdws.nn.Layer;
import org.sdws.nn.NetworkLayer;
import org.sdws.nn.NeuralNetwork;

import java.util.List;

public class Main {
    public static void main(String[] args) {
//        NNArray batch = NNArray.create(
//                List.of(
//                        List.of(1f, -1f, 0.5f)
//                )
//        );
//
//        NNArray singleInput = NNArray.create(
//                List.of(1f, 2f, 0f)
//        );
//
//        NNArray test = NNArray.createRandom(1, 20);
//
//        NNArray test2 = NNArray.createRandom(3, 4);
//        NNArray copy = test2.deepCopy();
//
//        System.out.println(test2);
//        System.out.println(copy);
//
//        System.out.println("-------------------------");
//        copy.set(0,0,999f);
//
//        System.out.println(test2);
//        System.out.println(copy);
//        System.out.println("-------------------------");
//        System.out.println("shape: " + test.shape);


//        Layer first = new Layer(2, 3, ActivationFunc.ReLU);
//        Layer last = new Layer(3, 2, ActivationFunc.Softmax);
//        //Layer out = new Layer(3, 3, ActivationFunc.Softmax);
//        NeuralNetwork network = new NeuralNetwork(
//                first, last
//        );
//
//        network.feedForward(test);
//        System.out.println("network final output: " + network.output());
//
//        NNArray oh = NNArray.create(List.of(0f, 1f));
//
//        NNArray ohBatch = NNArray.create(
//                List.of(
//                        List.of(0f, 0f, 1f),
//                        List.of(1f, 0f, 0f),
//                        List.of(0f, 1f, 0f),
//                        List.of(0f, 0f, 1f)
//                )
//        );
//        network.calculateCCELoss(oh);
//
//        System.out.println("calculated cce loss: " + network.loss);
//        System.out.println("calculated accuracy: " + network.calculateAccuracy(oh));

//        NNMatrix<Float> mat = NNMatrix.FromRows(
//                List.of(1f, 2f, 3f),
//                List.of(4f, 5f, 6f)
//        );
//
//        NNMatrix<Float> mat2 = NNMatrix.FromRows(
//                List.of(3f, 4f),
//                List.of(7f, 5f),
//                List.of(9f, 6f)
//        );
//
//        NArray<Float> tes1 = NArray.FromRows(
//                List.of(3f),
//                List.of(7f),
//                List.of(9f)
//        );
//
//
//        NArray<Float> tes2 = NArray.FromElements(1f, 2f, 3f);

//        System.out.println("obj: " + tes1 + ", dim: " + tes1.dimension + ", elements: " + tes1.elements);
//        System.out.println("obj: " + tes2 + ", dim: " + tes2.dimension + ", elements: " + tes2.elements);
//
//        NArray<Float> zeros = NArray.CreateZeros(Float.class, 5);
//        NArray<Float> tes3 = NArray.Mul(tes1, tes2);
//        System.out.println("obj: " + tes3 + ", dim: " + tes3.dimension + ", elements: " + tes3.elements);

        NetworkLayer testL = new NetworkLayer(3,4, ActivationFunc.ReLU);
        System.out.println("weights: " + testL.weights + "\nbiases: " +  testL.biases);
        System.out.println("weights shape: " + testL.weights.shape + "\nbiases shape: " +  testL.biases.shape);

        //System.out.println("tes2: " + tes2 + "\nshape: " +  tes2.shape);
        //testL.ForwardPass(tes2);

        NetworkLayer first = new NetworkLayer(3,4, ActivationFunc.ReLU);
        NetworkLayer second = new NetworkLayer(4,2, ActivationFunc.ReLU);
        NetworkLayer softmax = new NetworkLayer(2, 2, ActivationFunc.Softmax);
        NeuralNetwork network = new NeuralNetwork(
                first, second, softmax
        );

        NArray<Float> testInput = NArray.CreateRandom(3);
        System.out.println("testInput: " + testInput);

        NArray<Float> testOutput = NArray.FromElements(1f, 0f);

        network.feedForward(testInput);

        System.out.println("loss: " + network.loss);
        network.calculateLoss(testOutput, LossAlgorithm.CrossCatEntropy);

        System.out.println("nn output: " + network.output);
        System.out.println("loss: " + network.loss);
    }
}
