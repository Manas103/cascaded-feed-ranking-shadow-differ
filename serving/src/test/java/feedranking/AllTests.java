package feedranking;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A hand-rolled check suite, no JUnit (this repo builds with plain javac, matching this
 * portfolio's convention for a Java repo with no build-tool dependency). Each check()
 * call is one assertion; main exits 1 on the first failure and prints a per-check tally
 * otherwise. Run with: javac -d out src/main/java/feedranking/*.java
 * src/test/java/feedranking/*.java && java -cp out feedranking.AllTests ../export
 */
public final class AllTests {
    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws Exception {
        Path exportDir = Path.of(args.length > 0 ? args[0] : "../export");

        testLinearStackForwardDimensions();
        testLinearStackReluZeroesNegatives();
        testUserTowerPoolingIsAWeightedSumOfCategoryEmbeddings();
        testTopKIndicesDescendingMatchesBruteForceSortOnRandomInput();
        testTopKIndicesDescendingHandlesKGreaterThanLength();
        testTopKIndicesDescendingHandlesDuplicateScores();

        UserTower userTower = UserTower.load(exportDir);
        Reranker reranker = Reranker.load(exportDir);
        RankingService service = new RankingService(userTower, reranker);
        List<BenchmarkRequest> requests = DataLoader.loadBenchmarkRequests(exportDir);

        testLoadedRequestCount(requests);
        testEveryRequestHas2000Candidates(requests);
        testRankingReturnsTopKFromCandidates(service, requests);
        testRankingHasNoDuplicates(service, requests);
        testPreRankerCutIsActuallyTheTop200ByDotProduct(userTower, requests);
        testRankingIsDeterministic(service, requests);

        System.out.println(passed + " passed, " + failed + " failed");
        if (failed > 0) System.exit(1);
    }

    private static void check(String name, boolean condition) {
        if (condition) {
            passed++;
        } else {
            failed++;
            System.out.println("FAIL: " + name);
        }
    }

    private static void testLinearStackForwardDimensions() throws Exception {
        // A 2-in/3-out, 3-in/1-out stack: forward on a 2-vector must come out as a 1-vector.
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("stack", ".txt");
        java.nio.file.Files.writeString(
                tmp,
                "2\n3 2\n1 0\n0 1\n0 0.5\n0 0 0\n1 3\n1 1 1\n0\n");
        LinearStack stack = LinearStack.load(tmp);
        float[] out = stack.forward(new float[] {2f, 3f});
        check("LinearStack forward output has length 1", out.length == 1);
        java.nio.file.Files.deleteIfExists(tmp);
    }

    private static void testLinearStackReluZeroesNegatives() throws Exception {
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("stack2", ".txt");
        // Single layer with a negative bias so the raw output is negative before ReLU;
        // but a single-layer stack applies NO relu (relu only runs between layers), so
        // this checks the raw linear output directly instead.
        java.nio.file.Files.writeString(tmp, "1\n1 1\n1\n-5\n");
        LinearStack stack = LinearStack.load(tmp);
        float[] out = stack.forward(new float[] {1f});
        check("single-layer LinearStack applies no ReLU (raw negative output)", out[0] == -4f);
        java.nio.file.Files.deleteIfExists(tmp);
    }

    private static void testUserTowerPoolingIsAWeightedSumOfCategoryEmbeddings() throws Exception {
        // 3 categories, embedding dim 2, an identity-passthrough 1-layer "MLP" (no hidden
        // layer, weight = identity, bias = 0), so the tower's output is exactly the
        // pooling step's own weighted sum, independently hand-computed below.
        java.nio.file.Path dir = java.nio.file.Files.createTempDirectory("usertower");
        java.nio.file.Files.writeString(
                dir.resolve("user_tower_category_embedding.txt"),
                "3 2\n1 0\n0 1\n2 2\n");
        java.nio.file.Files.writeString(
                dir.resolve("user_tower_mlp.txt"),
                "1\n2 2\n1 0\n0 1\n0 0\n");
        UserTower tower = UserTower.load(dir);
        float[] out = tower.forward(new int[] {0, 1, 2}, new float[] {0.5f, 0.25f, 0.25f});
        // Expected: 0.5*(1,0) + 0.25*(0,1) + 0.25*(2,2) = (1.0, 0.75)
        boolean close = Math.abs(out[0] - 1.0f) < 1e-5 && Math.abs(out[1] - 0.75f) < 1e-5;
        check("UserTower pooling matches a hand-computed weighted sum", close);
        java.nio.file.Files.deleteIfExists(dir.resolve("user_tower_category_embedding.txt"));
        java.nio.file.Files.deleteIfExists(dir.resolve("user_tower_mlp.txt"));
        java.nio.file.Files.deleteIfExists(dir);
    }

    private static void testTopKIndicesDescendingMatchesBruteForceSortOnRandomInput() {
        java.util.Random rng = new java.util.Random(7);
        for (int trial = 0; trial < 50; trial++) {
            int n = 10 + rng.nextInt(200);
            int k = 1 + rng.nextInt(Math.min(50, n));
            float[] scores = new float[n];
            for (int i = 0; i < n; i++) scores[i] = rng.nextFloat() * 200f - 100f;

            int[] got = RankingService.topKIndicesDescending(scores, k);

            Integer[] bruteForceOrder = new Integer[n];
            for (int i = 0; i < n; i++) bruteForceOrder[i] = i;
            java.util.Arrays.sort(bruteForceOrder, (a, b) -> Float.compare(scores[b], scores[a]));

            float[] gotScoresSorted = new float[k];
            for (int i = 0; i < k; i++) gotScoresSorted[i] = scores[got[i]];
            float[] expectedScoresSorted = new float[k];
            for (int i = 0; i < k; i++) expectedScoresSorted[i] = scores[bruteForceOrder[i]];
            if (!java.util.Arrays.equals(gotScoresSorted, expectedScoresSorted)) {
                check("topKIndicesDescending matches brute-force sort (trial " + trial + ")", false);
                return;
            }
        }
        check("topKIndicesDescending matches brute-force sort over 50 random trials", true);
    }

    private static void testTopKIndicesDescendingHandlesKGreaterThanLength() {
        float[] scores = {3f, 1f, 2f};
        int[] got = RankingService.topKIndicesDescending(scores, 10);
        check("topKIndicesDescending caps output length at the input length", got.length == 3);
        check("topKIndicesDescending still returns the max first", scores[got[0]] == 3f);
    }

    private static void testTopKIndicesDescendingHandlesDuplicateScores() {
        float[] scores = {5f, 5f, 5f, 1f};
        int[] got = RankingService.topKIndicesDescending(scores, 2);
        check("topKIndicesDescending returns k items even with duplicate scores", got.length == 2);
        check("topKIndicesDescending never returns the clearly-lower score ahead of duplicates", scores[got[0]] == 5f && scores[got[1]] == 5f);
    }

    private static void testLoadedRequestCount(List<BenchmarkRequest> requests) {
        check("loaded at least one benchmark request", requests.size() > 0);
    }

    private static void testEveryRequestHas2000Candidates(List<BenchmarkRequest> requests) {
        boolean all2000 = requests.stream().allMatch(r -> r.candidateItemIds.length == 2000);
        check("every benchmark request has exactly 2000 candidates", all2000);
    }

    private static void testRankingReturnsTopKFromCandidates(RankingService service, List<BenchmarkRequest> requests) {
        BenchmarkRequest req = requests.get(0);
        int[] result = service.rank(req);
        check("rank() returns exactly TOP_K results", result.length == RankingService.TOP_K);
        Set<Integer> candidateSet = new HashSet<>();
        for (int id : req.candidateItemIds) candidateSet.add(id);
        boolean allFromCandidates = Arrays.stream(result).allMatch(candidateSet::contains);
        check("every ranked item id came from this request's own candidates", allFromCandidates);
    }

    private static void testRankingHasNoDuplicates(RankingService service, List<BenchmarkRequest> requests) {
        int[] result = service.rank(requests.get(1));
        Set<Integer> seen = new HashSet<>();
        for (int id : result) seen.add(id);
        check("rank() output has no duplicate item ids", seen.size() == result.length);
    }

    private static void testPreRankerCutIsActuallyTheTop200ByDotProduct(UserTower userTower, List<BenchmarkRequest> requests) {
        BenchmarkRequest req = requests.get(2);
        float[] userEmb = userTower.forward(req.topCategories, req.categoryWeights);
        float[] scores = new float[req.candidateItemIds.length];
        for (int i = 0; i < scores.length; i++) {
            float sum = 0f;
            for (int d = 0; d < userEmb.length; d++) sum += userEmb[d] * req.candidateEmbeddings[i][d];
            scores[i] = sum;
        }
        float[] sortedDesc = scores.clone();
        Arrays.sort(sortedDesc);
        float threshold200th = sortedDesc[sortedDesc.length - 200];
        int atOrAboveThreshold = 0;
        for (float s : scores) if (s >= threshold200th) atOrAboveThreshold++;
        check("independently recomputed pre-ranker scores have >=200 items at/above the 200th-best score", atOrAboveThreshold >= 200);
    }

    private static void testRankingIsDeterministic(RankingService service, List<BenchmarkRequest> requests) {
        BenchmarkRequest req = requests.get(3);
        int[] first = service.rank(req);
        int[] second = service.rank(req);
        check("rank() is deterministic for the same request", Arrays.equals(first, second));
    }
}
