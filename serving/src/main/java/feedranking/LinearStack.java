package feedranking;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * A stack of Linear layers with ReLU between every pair but the last, ported by hand from
 * the PyTorch state dict scripts/export_for_java.py wrote out. No matrix library: each
 * layer is a plain float[][] weight and float[] bias, and forward() is the same
 * multiply-add a real inference runtime would do, just written out explicitly.
 */
public final class LinearStack {
    private final float[][][] weights;
    private final float[][] biases;

    private LinearStack(float[][][] weights, float[][] biases) {
        this.weights = weights;
        this.biases = biases;
    }

    public static LinearStack load(Path path) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            int numLayers = Integer.parseInt(reader.readLine().trim());
            float[][][] weights = new float[numLayers][][];
            float[][] biases = new float[numLayers][];
            for (int layer = 0; layer < numLayers; layer++) {
                String[] dims = reader.readLine().trim().split("\\s+");
                int rows = Integer.parseInt(dims[0]);
                int cols = Integer.parseInt(dims[1]);
                float[][] w = new float[rows][cols];
                for (int r = 0; r < rows; r++) {
                    w[r] = parseFloats(reader.readLine(), cols);
                }
                float[] b = parseFloats(reader.readLine(), rows);
                weights[layer] = w;
                biases[layer] = b;
            }
            return new LinearStack(weights, biases);
        }
    }

    private static float[] parseFloats(String line, int expectedLength) {
        String[] parts = line.trim().split("\\s+");
        float[] out = new float[expectedLength];
        for (int i = 0; i < expectedLength; i++) out[i] = Float.parseFloat(parts[i]);
        return out;
    }

    /** Runs every layer in order, applying ReLU after every layer except the last. */
    public float[] forward(float[] input) {
        float[] x = input;
        for (int layer = 0; layer < weights.length; layer++) {
            x = linear(x, weights[layer], biases[layer]);
            if (layer < weights.length - 1) relu(x);
        }
        return x;
    }

    /**
     * Runs the whole stack over a batch of inputs at once, row-major over the weight
     * matrix in the outer loop so each weight row ({@code wr}) is read once and reused
     * across every sample in the batch, instead of re-reading the full weight matrix
     * once per sample the way {@link #forward} called 200 times in a loop would. Same
     * total multiply-adds as 200 separate {@link #forward} calls; the measured speed
     * difference (see the README's Findings section) is purely from cache reuse and
     * fewer, larger loop bodies for the JIT to work with, not a different algorithm.
     */
    public float[][] forwardBatch(float[][] inputs) {
        float[][] x = inputs;
        for (int layer = 0; layer < weights.length; layer++) {
            x = linearBatch(x, weights[layer], biases[layer]);
            if (layer < weights.length - 1) {
                for (float[] row : x) relu(row);
            }
        }
        return x;
    }

    private static float[] linear(float[] x, float[][] w, float[] b) {
        int rows = w.length;
        int cols = w[0].length;
        float[] out = new float[rows];
        for (int r = 0; r < rows; r++) {
            float sum = b[r];
            float[] wr = w[r];
            for (int c = 0; c < cols; c++) sum += wr[c] * x[c];
            out[r] = sum;
        }
        return out;
    }

    private static float[][] linearBatch(float[][] inputs, float[][] w, float[] b) {
        int batch = inputs.length;
        int rows = w.length;
        int cols = w[0].length;
        float[][] out = new float[batch][rows];
        for (int r = 0; r < rows; r++) {
            float[] wr = w[r];
            float br = b[r];
            for (int i = 0; i < batch; i++) {
                float[] xi = inputs[i];
                float sum = br;
                for (int c = 0; c < cols; c++) sum += wr[c] * xi[c];
                out[i][r] = sum;
            }
        }
        return out;
    }

    private static void relu(float[] x) {
        for (int i = 0; i < x.length; i++) if (x[i] < 0f) x[i] = 0f;
    }
}
