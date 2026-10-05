package feedranking;

/**
 * The serving-time cascade: compute the user embedding once, score every retrieval
 * candidate with the pre-ranker's cached-embedding dot product, keep the top 200, and run
 * only those 200 through the heavier reranker. This is the exact tradeoff
 * measure_ndcg_and_compute.py measures in Python; RankingService is the same cascade
 * ported to the language the p99/QPS benchmark actually runs in.
 *
 * <p>Top-k selection ({@link #topKIndicesDescending}) is a bounded primitive min-heap, not
 * a full sort: keeping only 200 of 2,000 candidates with a boxed {@code Integer[]} and a
 * lambda comparator (the first version of this file) spends most of its time on
 * autoboxing and comparator dispatch rather than the actual ranking math, which is the
 * real bottleneck a production top-k path avoids the same way. See the README's Findings
 * section for the measured difference.
 */
public final class RankingService {
    public static final int PRE_RANKER_CUT = 200;
    public static final int TOP_K = 10;

    private final UserTower userTower;
    private final Reranker reranker;

    public RankingService(UserTower userTower, Reranker reranker) {
        this.userTower = userTower;
        this.reranker = reranker;
    }

    public int[] rank(BenchmarkRequest request) {
        float[] userEmb = userTower.forward(request.topCategories, request.categoryWeights);
        int n = request.candidateItemIds.length;

        float[] preScores = new float[n];
        for (int i = 0; i < n; i++) preScores[i] = dot(userEmb, request.candidateEmbeddings[i]);

        int[] survivorIndices = topKIndicesDescending(preScores, Math.min(PRE_RANKER_CUT, n));
        int survivorCount = survivorIndices.length;
        int[] survivorItemIds = new int[survivorCount];
        float[][] survivorEmbeddings = new float[survivorCount][];
        float[] survivorQuality = new float[survivorCount];
        for (int i = 0; i < survivorCount; i++) {
            int idx = survivorIndices[i];
            survivorItemIds[i] = request.candidateItemIds[idx];
            survivorEmbeddings[i] = request.candidateEmbeddings[idx];
            survivorQuality[i] = request.candidateQuality[idx];
        }
        float[] rerankScores = reranker.scoreBatch(userEmb, survivorEmbeddings, survivorQuality);

        int[] rerankOrder = topKIndicesDescending(rerankScores, Math.min(TOP_K, survivorCount));
        int[] result = new int[rerankOrder.length];
        for (int i = 0; i < result.length; i++) result[i] = survivorItemIds[rerankOrder[i]];
        return result;
    }

    private static float dot(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    /** Indices of the top k scores, descending, via a bounded min-heap: O(n log k), no
     * boxing. Returns min(k, scores.length) indices. */
    static int[] topKIndicesDescending(float[] scores, int k) {
        int n = scores.length;
        k = Math.min(k, n);
        float[] heapScore = new float[k];
        int[] heapIndex = new int[k];
        int size = 0;
        for (int i = 0; i < n; i++) {
            float s = scores[i];
            if (size < k) {
                heapScore[size] = s;
                heapIndex[size] = i;
                size++;
                if (size == k) {
                    for (int p = k / 2 - 1; p >= 0; p--) siftDown(heapScore, heapIndex, p, k);
                }
            } else if (s > heapScore[0]) {
                heapScore[0] = s;
                heapIndex[0] = i;
                siftDown(heapScore, heapIndex, 0, k);
            }
        }
        if (size < k) {
            for (int p = size / 2 - 1; p >= 0; p--) siftDown(heapScore, heapIndex, p, size);
        }
        int[] result = new int[size];
        int remaining = size;
        while (remaining > 0) {
            result[remaining - 1] = heapIndex[0];
            remaining--;
            heapScore[0] = heapScore[remaining];
            heapIndex[0] = heapIndex[remaining];
            siftDown(heapScore, heapIndex, 0, remaining);
        }
        return result;
    }

    private static void siftDown(float[] score, int[] idx, int pos, int size) {
        while (true) {
            int left = 2 * pos + 1;
            int right = 2 * pos + 2;
            int smallest = pos;
            if (left < size && score[left] < score[smallest]) smallest = left;
            if (right < size && score[right] < score[smallest]) smallest = right;
            if (smallest == pos) break;
            float ts = score[pos];
            score[pos] = score[smallest];
            score[smallest] = ts;
            int ti = idx[pos];
            idx[pos] = idx[smallest];
            idx[smallest] = ti;
            pos = smallest;
        }
    }
}
