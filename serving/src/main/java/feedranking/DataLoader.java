package feedranking;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Parses scripts/export_for_java.py's plain-text export: benchmark_requests.tsv (one
 * request per line) joined against candidate_embeddings.tsv (one candidate item per
 * line, keyed by request index) on the request index column both files share. */
public final class DataLoader {
    private DataLoader() {}

    public static List<BenchmarkRequest> loadBenchmarkRequests(Path exportDir) throws IOException {
        Map<Integer, int[]> itemIdsByRequest = new HashMap<>();
        Map<Integer, Integer> userIdByRequest = new HashMap<>();
        Map<Integer, int[]> topCategoriesByRequest = new HashMap<>();
        Map<Integer, float[]> categoryWeightsByRequest = new HashMap<>();

        try (BufferedReader reader = Files.newBufferedReader(exportDir.resolve("benchmark_requests.tsv"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] cols = line.split("\t");
                int requestIndex = Integer.parseInt(cols[0]);
                int userId = Integer.parseInt(cols[1]);
                int[] topCategories = parseInts(cols[2]);
                float[] categoryWeights = parseFloats(cols[3]);
                int[] candidateItemIds = parseInts(cols[4]);
                userIdByRequest.put(requestIndex, userId);
                topCategoriesByRequest.put(requestIndex, topCategories);
                categoryWeightsByRequest.put(requestIndex, categoryWeights);
                itemIdsByRequest.put(requestIndex, candidateItemIds);
            }
        }

        Map<Integer, Map<Integer, Integer>> positionOfItem = new HashMap<>();
        Map<Integer, float[][]> embeddingsByRequest = new HashMap<>();
        Map<Integer, float[]> qualityByRequest = new HashMap<>();
        for (Map.Entry<Integer, int[]> e : itemIdsByRequest.entrySet()) {
            int n = e.getValue().length;
            embeddingsByRequest.put(e.getKey(), new float[n][]);
            qualityByRequest.put(e.getKey(), new float[n]);
            Map<Integer, Integer> pos = new HashMap<>();
            for (int i = 0; i < n; i++) pos.put(e.getValue()[i], i);
            positionOfItem.put(e.getKey(), pos);
        }

        try (BufferedReader reader = Files.newBufferedReader(exportDir.resolve("candidate_embeddings.tsv"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] cols = line.split("\t");
                int requestIndex = Integer.parseInt(cols[0]);
                int itemId = Integer.parseInt(cols[1]);
                float quality = Float.parseFloat(cols[2]);
                float[] emb = parseFloats(cols[3]);
                int position = positionOfItem.get(requestIndex).get(itemId);
                embeddingsByRequest.get(requestIndex)[position] = emb;
                qualityByRequest.get(requestIndex)[position] = quality;
            }
        }

        List<BenchmarkRequest> requests = new ArrayList<>();
        for (Integer requestIndex : userIdByRequest.keySet()) {
            requests.add(
                    new BenchmarkRequest(
                            userIdByRequest.get(requestIndex),
                            topCategoriesByRequest.get(requestIndex),
                            categoryWeightsByRequest.get(requestIndex),
                            itemIdsByRequest.get(requestIndex),
                            embeddingsByRequest.get(requestIndex),
                            qualityByRequest.get(requestIndex)));
        }
        return requests;
    }

    private static int[] parseInts(String s) {
        String[] parts = s.trim().split("\\s+");
        int[] out = new int[parts.length];
        for (int i = 0; i < parts.length; i++) out[i] = Integer.parseInt(parts[i]);
        return out;
    }

    private static float[] parseFloats(String s) {
        String[] parts = s.trim().split("\\s+");
        float[] out = new float[parts.length];
        for (int i = 0; i < parts.length; i++) out[i] = Float.parseFloat(parts[i]);
        return out;
    }
}
