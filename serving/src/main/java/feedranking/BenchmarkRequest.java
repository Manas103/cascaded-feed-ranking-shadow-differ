package feedranking;

/**
 * One precomputed query: a user's raw tower inputs plus its 2,000-item retrieval
 * candidate set with cached embeddings and quality scalars, exactly what a real request
 * would carry in from retrieval and a Redis embedding cache (see RankingService and
 * serving/README.md).
 */
public final class BenchmarkRequest {
    public final int userId;
    public final int[] topCategories;
    public final float[] categoryWeights;
    public final int[] candidateItemIds;
    public final float[][] candidateEmbeddings;
    public final float[] candidateQuality;

    public BenchmarkRequest(
            int userId,
            int[] topCategories,
            float[] categoryWeights,
            int[] candidateItemIds,
            float[][] candidateEmbeddings,
            float[] candidateQuality) {
        this.userId = userId;
        this.topCategories = topCategories;
        this.categoryWeights = categoryWeights;
        this.candidateItemIds = candidateItemIds;
        this.candidateEmbeddings = candidateEmbeddings;
        this.candidateQuality = candidateQuality;
    }
}
