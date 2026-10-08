package org.sdws.visualizer;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.Stage;

public class MainVisualizer extends Application {

    public static void main(String[] args) {
        launch(args);
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
