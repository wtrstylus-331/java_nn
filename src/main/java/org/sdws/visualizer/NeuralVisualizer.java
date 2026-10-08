package org.sdws.visualizer;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import org.sdws.nn.NeuralNetwork;

public class NeuralVisualizer extends Application {
    protected static NeuralNetwork neuralNetwork = null;

//    public static void main(String[] args) {
//        launch(args);
//    }
    public static void run(NeuralNetwork network) {
        neuralNetwork = network;
        launch(NeuralVisualizer.class);
    }


    @Override
    public void start(Stage primaryStage) {
        VBox root = new VBox();
        Text title = new Text("Visualizer");
        root.getChildren().add(title);

        Scene s = new Scene(root, 800, 600);

        primaryStage.setScene(s);
        primaryStage.setTitle("Visualizer");
        primaryStage.show();
    }
}
