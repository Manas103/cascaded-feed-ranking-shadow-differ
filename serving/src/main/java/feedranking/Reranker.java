package feedranking;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Ports reranker.py's cross_features + forward: [userEmb, itemEmb, userEmb*itemEmb,
 * quality] through the 3-layer MLP exported to reranker_mlp.txt. This is the "heavier"
 * tier BenchmarkMain pays for only on the 200 pre-ranker survivors, never on all 2,000
 * candidates.
 */
public final class Reranker {
    private final LinearStack mlp;

    private Reranker(LinearStack mlp) {
        this.mlp = mlp;
    }

    public static Reranker load(Path exportDir) throws IOException {
        return new Reranker(LinearStack.load(exportDir.resolve("reranker_mlp.txt")));
    }

    public float score(float[] userEmb, float[] itemEmb, float quality) {
        return mlp.forward(crossFeatures(userEmb, itemEmb, quality))[0];
    }

    /** Scores every item in one batched forward pass; see LinearStack.forwardBatch. */
    public float[] scoreBatch(float[] userEmb, float[][] itemEmbs, float[] quality) {
        float[][] features = new float[itemEmbs.length][];
        for (int i = 0; i < itemEmbs.length; i++) {
            features[i] = crossFeatures(userEmb, itemEmbs[i], quality[i]);
        }
        float[][] outputs = mlp.forwardBatch(features);
        float[] scores = new float[outputs.length];
        for (int i = 0; i < outputs.length; i++) scores[i] = outputs[i][0];
        return scores;
    }

    private static float[] crossFeatures(float[] userEmb, float[] itemEmb, float quality) {
        int d = userEmb.length;
        float[] features = new float[3 * d + 1];
        System.arraycopy(userEmb, 0, features, 0, d);
        System.arraycopy(itemEmb, 0, features, d, d);
        for (int i = 0; i < d; i++) features[2 * d + i] = userEmb[i] * itemEmb[i];
        features[3 * d] = quality;
        return features;
    }
}
