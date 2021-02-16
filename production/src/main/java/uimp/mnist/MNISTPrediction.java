package uimp.mnist;

import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.logging.Level;
import java.util.logging.Logger;

public class MNISTPrediction {

    private static final Logger logger = Logger.getLogger(MNISTPrediction.class.getName());

    /**
     * Zeros the supplied array.
     *
     * @param data The array to zero.
     */
    public static void zeroData(float[][][][] data) {
        // Zero the array
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                for (int k = 0; k < data[i][j].length; k++) {
                    Arrays.fill(data[i][j][k], 0.0f);
                }
            }
        }
    }

    /**
     * Writes out sparse data into the last two dimensions of the supplied 4d array.
     *
     * @param data The 4d array to write to.
     * @param image Is the image value in level of grey [0, 1]
     */
    public static void writeData(float[][][][] data, float[][] image) {
        zeroData(data);

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                data[0][0][i][j] = image[i][j];
                //data[0][0][i][j] = (image[i][j] - 0.1307f) / 0.3081f;
            }
        }
    }


    /**
     * Find the maximum probability and return it's index.
     *
     * @param probabilities The probabilites.
     * @return The index of the max.
     */
    public static int pred(float[] probabilities) {
        float maxVal = Float.NEGATIVE_INFINITY;
        int idx = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i];
                idx = i;
            }
        }
        return idx;
    }

    public static void main(String[] args) throws OrtException, IOException {
        if (args.length != 2) {
            System.out.println("Usage: MNISTPrediction <model-path> <test-image>");
            return;
        }

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession.SessionOptions opts = new SessionOptions()) {

            opts.setOptimizationLevel(OptLevel.BASIC_OPT);

            logger.info("Loading model from " + args[0]);
            try (OrtSession session = env.createSession(args[0], opts)) {

                logger.info("Inputs:");
                for (NodeInfo i : session.getInputInfo().values()) {
                    logger.info(i.toString());
                }

                logger.info("Outputs:");
                for (NodeInfo i : session.getOutputInfo().values()) {
                    logger.info(i.toString());
                }

                String inputName = session.getInputNames().iterator().next();

                float[][][][] testData = new float[1][1][32][32];
                float[][] image = new float[32][32];
                BufferedImage img = null;

                try{
                    File f = new File(args[1]);
                    img = ImageIO.read(f);
                }catch(IOException e){
                    System.out.println(e);
                    System.exit(0);
                }

                int height = img.getHeight();
                int width = img.getWidth();
                Raster raster = img.getData();
                for(int i = 0; i < height; i++){
                    for(int j = 0; j < width; j++){

                        image[i][j] = 1-raster.getSample(j,i,0)/255;
                    }
                }

                JPanel panel = new JPanel();
                panel.setSize(250,320);
                panel.setBackground(Color.CYAN);
                ImageIcon icon = new ImageIcon(args[1]);
                JLabel label = new JLabel();
                label.setIcon(icon);
                panel.add(label);
                JFrame frame = new JFrame();
                frame.getContentPane().add(panel);
                frame.setSize(250,320);
                frame.setVisible(true);

                writeData(testData, image);

                try (OnnxTensor test = OnnxTensor.createTensor(env, testData);
                     Result output = session.run(Collections.singletonMap(inputName, test))) {

                    float[][] outputProbs = (float[][]) output.get(0).getValue();
                    int predLabel = pred(outputProbs[0]);


                    logger.log(Level.INFO, "Predicted label = " + predLabel);
                }
            }
        }

        logger.info("Done!");
    }
}
