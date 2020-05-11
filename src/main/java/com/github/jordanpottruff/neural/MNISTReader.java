package com.github.jordanpottruff.neural;

import com.github.jordanpottruff.jgml.VecN;
import com.github.jordanpottruff.neural.data.DataSet;
import com.github.jordanpottruff.neural.data.Observation;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Provides functionality for reading the Modified National Institute of Standards and Technology (MNIST) database of
 * handwritten digits.
 */
public class MNISTReader {

    public static final int IMAGE_SIZE = 28;

    private final String imageFilename;
    private final String labelFilename;

    /**
     * Creates a MNIST reader. Requires the file names for the image and label files.
     *
     * @param imageFilename the filename for the image file.
     * @param labelFilename the filename for the label file.
     */
    public MNISTReader(String imageFilename, String labelFilename) {
        this.imageFilename = imageFilename;
        this.labelFilename = labelFilename;
    }

    /**
     * Reads the MNIST files and returns a list of the image/label combinations as an MNISTImage.
     *
     * @return the list of MNIST image/label combinations.
     */
    public List<MNISTImage> getImages() {
        ArrayList<MNISTImage> images = new ArrayList<>();
        try {
            FileInputStream imageStream = new FileInputStream(imageFilename);
            FileInputStream labelStream = new FileInputStream(labelFilename);

            // Read image stream headers.
            readNextInt(imageStream); // Magic number
            int numberOfImages = readNextInt(imageStream);
            readNextInt(imageStream); // Number of rows
            readNextInt(imageStream); // Number of columns

            // Read label stream headers.
            readNextInt(labelStream); // Magic number
            readNextInt(labelStream); // Number of labels

            for (int i = 0; i < numberOfImages; i++) {
                int label = labelStream.read();
                ArrayList<Double> image = new ArrayList<>();
                for (int p = 0; p < IMAGE_SIZE * IMAGE_SIZE; p++) {
                    image.add(imageStream.read() / 255.0);
                }
                images.add(new MNISTImage(image, label));
            }
            return images;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    // Reads and returns 32 bits of the input stream.
    private int readNextInt(FileInputStream input) throws IOException {
        return (input.read() << 24) | (input.read() << 16) | (input.read() << 8) | (input.read());
    }

    /**
     * Reads the MNIST files and returns a list of the image/label combinations as a DataSet.
     *
     * @return the DataSet of MNIST images.
     */
    public DataSet getImageDataSet() {
        String[] classes = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        List<Observation> obs = getImages().stream().map(MNISTImage::toObservation).collect(Collectors.toList());
        return new DataSet(obs, classes);
    }

    /**
     * Defines single images in the MNIST database.
     */
    public static class MNISTImage {

        private final List<Double> image;
        private final int label;

        /**
         * Creates an MNIST image object out of pixel data and a label.
         *
         * @param image the values of the pixels (each between 0-1), going top-down, left-right.
         * @param label the label of the image (i.e. the digit).
         */
        public MNISTImage(List<Double> image, int label) {
            this.image = image;
            this.label = label;
        }

        /**
         * Converts the MNIST image object to a vector representation.
         *
         * @return a VecN storing the pixel information of the image, going top-down, left-right.
         */
        public VecN toVecN() {
            double[] imageArray = new double[image.size()];
            for(int i=0; i<imageArray.length; i++) {
                imageArray[i] = image.get(i);
            }
            return new VecN(imageArray);
        }

        /**
         * Converts the MNIST image object to an Observation.
         *
         * @return an Observation storing the pixel information as the attributes and the label as the classifier.
         */
        public Observation toObservation() {
            return new Observation(toVecN(), Integer.toString(label));
        }

        /**
         * Returns a string representation showing the image data and label.
         *
         * @return string representation of the MNIST image.
         */
        public String toString() {
            StringBuilder str = new StringBuilder("IMAGE:");
            for (int p = 0; p < IMAGE_SIZE * IMAGE_SIZE; p++) {
                if (p % IMAGE_SIZE == 0) {
                    str.append("\n");
                }
                char symbol = image.get(p) >= 0.25 ? 'X' : '.';
                str.append(symbol).append("  ");
            }
            str.append("\nLABEL: ").append(label);
            return str.toString();
        }
    }

}
