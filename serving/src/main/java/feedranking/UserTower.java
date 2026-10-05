package feedranking;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Ports towers.py's UserTower forward pass: a weighted sum over the caller's top category
 * embeddings (nn.EmbeddingBag(mode="sum") with per_sample_weights, done by hand here as a
 * plain weighted accumulation), then the same 2-layer MLP exported to user_tower_mlp.txt.
 */
public final class UserTower {
    private final float[][] categoryEmbedding;
    private final LinearStack mlp;

    private UserTower(float[][] categoryEmbedding, LinearStack mlp) {
        this.categoryEmbedding = categoryEmbedding;
        this.mlp = mlp;
    }

    public static UserTower load(Path exportDir) throws IOException {
        float[][] categoryEmbedding = loadMatrix(exportDir.resolve("user_tower_category_embedding.txt"));
        LinearStack mlp = LinearStack.load(exportDir.resolve("user_tower_mlp.txt"));
        return new UserTower(categoryEmbedding, mlp);
    }

    private static float[][] loadMatrix(Path path) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String[] dims = reader.readLine().trim().split("\\s+");
            int rows = Integer.parseInt(dims[0]);
            int cols = Integer.parseInt(dims[1]);
            float[][] m = new float[rows][cols];
            for (int r = 0; r < rows; r++) {
                String[] parts = reader.readLine().trim().split("\\s+");
                for (int c = 0; c < cols; c++) m[r][c] = Float.parseFloat(parts[c]);
            }
            return m;
        }
    }

    public float[] forward(int[] topCategories, float[] categoryWeights) {
        int embDim = categoryEmbedding[0].length;
        float[] pooled = new float[embDim];
        for (int i = 0; i < topCategories.length; i++) {
            float[] catEmb = categoryEmbedding[topCategories[i]];
            float w = categoryWeights[i];
            for (int d = 0; d < embDim; d++) pooled[d] += w * catEmb[d];
        }
        return mlp.forward(pooled);
    }
}
