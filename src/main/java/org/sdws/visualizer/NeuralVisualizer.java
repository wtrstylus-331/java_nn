package org.sdws.visualizer;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.geometry.Rectangle2D;
import javafx.scene.Scene;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.Background;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;
import javafx.stage.Screen;
import javafx.stage.Stage;
import org.sdws.nn.NetworkLayer;
import org.sdws.nn.NeuralNetwork;

import java.util.ArrayList;

public class NeuralVisualizer extends Application {
    protected static NeuralNetwork neuralNetwork = null;
    private static VBox[] layerBoxes;

    private static double WIDTH = 0;
    private static double HEIGHT = 0;

    public static void run(NeuralNetwork network) {
        neuralNetwork = network;
        launch(NeuralVisualizer.class);
    }

    @Override
    public void start(Stage primaryStage) {
        if (neuralNetwork == null) {
            System.err.println("Neural Network is not assigned for visualization");
            return;
        }

        initBounds();

        double netDiagramHeight = HEIGHT - (HEIGHT * 0.3);
        layerBoxes = new VBox[neuralNetwork.getLayers().size()];

        ArrayList<NetworkLayer> layers = neuralNetwork.getLayers();
        for (NetworkLayer layer : layers) {
            System.out.println(layer.getNeurons());
        }

        ScrollPane scrollPane = new ScrollPane();
        scrollPane.setPannable(true);
        scrollPane.setHbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        scrollPane.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        scrollPane.setBackground(Background.EMPTY);
        scrollPane.setPrefHeight(netDiagramHeight);

        HBox layerBox = new HBox();
        layerBox.setSpacing(120);
        layerBox.setAlignment(Pos.CENTER);
        layerBox.setBackground(Background.EMPTY);

        StackPane holder = new StackPane();
        holder.setAlignment(Pos.CENTER);
        holder.setBackground(Background.EMPTY);
        holder.getChildren().add(layerBox);
        holder.setPadding(new Insets(
                HEIGHT - (HEIGHT * 0.95),
                WIDTH - (WIDTH * 0.9),
                HEIGHT - (HEIGHT * 0.95),
                WIDTH - (WIDTH * 0.9)));

        scrollPane.setContent(holder);
        scrollPane.setFitToHeight(true);
        scrollPane.setFitToWidth(true);

        VBox inputLayer = new VBox();
        inputLayer.setAlignment(Pos.CENTER);
        inputLayer.setSpacing(30);

        for (int i = 0; i < neuralNetwork.getInput().getVecArray().length; i++) {
            Rectangle rectangle = new Rectangle();
            rectangle.setFill(Color.GREEN);
            rectangle.setWidth(35);
            rectangle.setHeight(25);
            inputLayer.getChildren().add(rectangle);
        }

        layerBox.getChildren().add(inputLayer);


        for (int i = 0; i < layers.size(); i++) {
            VBox neuronBox = new VBox();
            neuronBox.setAlignment(Pos.CENTER);
            neuronBox.setSpacing(30);

            for (int j = 0; j < layers.get(i).getNeurons(); j++) {
                Circle circle = new Circle();
                circle.setFill(Color.GREEN);
                circle.setRadius(20);
                neuronBox.getChildren().add(circle);
            }
            System.out.println(layers.get(i));
            layerBox.getChildren().add(neuronBox);
        }

        StackPane networkDiagram = new StackPane();
        networkDiagram.setStyle("-fx-background-color: transparent;");
        networkDiagram.getChildren().add(scrollPane);
        networkDiagram.setMaxHeight(netDiagramHeight);
        networkDiagram.setPrefHeight(netDiagramHeight);

        VBox root = new VBox();
        root.setStyle("-fx-background-color: navy;");
        root.getChildren().addAll(networkDiagram);

        Scene main = new Scene(root);

        primaryStage.setScene(main);
        primaryStage.setTitle("Visualizer");
        primaryStage.setMaximized(true);
        primaryStage.show();
    }

    private static void initBounds() {
        Rectangle2D visualBounds = Screen.getPrimary().getVisualBounds();
        WIDTH = visualBounds.getWidth();
        HEIGHT = visualBounds.getHeight();
    }
}
