package feedranking;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Closed-loop load benchmark: THREADS worker threads, each in a tight loop ranking one of
 * the 50 precomputed benchmark requests (cycling, striped by thread id so no two threads
 * start on the same request) for DURATION_SECONDS, recording every request's latency. QPS
 * is total completed requests over wall-clock duration; p99 is read off the full sorted
 * latency array, not estimated. THREADS defaults to 8 to match "one 8-core host"; pass a
 * different value as argv[0] to run at a different concurrency.
 */
public final class BenchmarkMain {
    private static final int DURATION_SECONDS = 10;
    private static final int WARMUP_REQUESTS_PER_THREAD = 200;

    public static void main(String[] args) throws Exception {
        int threads = args.length > 0 ? Integer.parseInt(args[0]) : 8;
        Path exportDir = Path.of(args.length > 1 ? args[1] : "../export");

        UserTower userTower = UserTower.load(exportDir);
        Reranker reranker = Reranker.load(exportDir);
        RankingService service = new RankingService(userTower, reranker);
        List<BenchmarkRequest> requests = DataLoader.loadBenchmarkRequests(exportDir);
        System.err.println("loaded " + requests.size() + " benchmark requests");

        // Warm up every worker's JIT path before any timing starts.
        for (int i = 0; i < WARMUP_REQUESTS_PER_THREAD * threads; i++) {
            service.rank(requests.get(i % requests.size()));
        }

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        AtomicBoolean stop = new AtomicBoolean(false);
        AtomicLong totalCompleted = new AtomicLong(0);
        List<long[]> latencyArrays = new ArrayList<>();
        List<int[]> countHolders = new ArrayList<>();

        int maxPerThread = 2_000_000;
        List<Runnable> workers = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            long[] latencies = new long[maxPerThread];
            int[] countHolder = new int[1];
            latencyArrays.add(latencies);
            countHolders.add(countHolder);
            int threadId = t;
            workers.add(() -> {
                int cursor = threadId;
                int count = 0;
                while (!stop.get() && count < maxPerThread) {
                    BenchmarkRequest req = requests.get(cursor % requests.size());
                    long start = System.nanoTime();
                    service.rank(req);
                    long elapsed = System.nanoTime() - start;
                    latencies[count++] = elapsed;
                    cursor += threads;
                }
                countHolder[0] = count;
                totalCompleted.addAndGet(count);
            });
        }

        long wallStart = System.nanoTime();
        for (Runnable w : workers) pool.submit(w);
        Thread.sleep(DURATION_SECONDS * 1000L);
        stop.set(true);
        pool.shutdown();
        pool.awaitTermination(30, TimeUnit.SECONDS);
        long wallElapsedNanos = System.nanoTime() - wallStart;

        int total = 0;
        for (int[] c : countHolders) total += c[0];
        long[] all = new long[total];
        int offset = 0;
        for (int t = 0; t < threads; t++) {
            int count = countHolders.get(t)[0];
            System.arraycopy(latencyArrays.get(t), 0, all, offset, count);
            offset += count;
        }
        Arrays.sort(all);

        double qps = total / (wallElapsedNanos / 1e9);
        long p50 = all[(int) (all.length * 0.50)];
        long p99 = all[(int) (all.length * 0.99)];
        long max = all[all.length - 1];

        System.out.println("threads=" + threads);
        System.out.println("duration_seconds=" + DURATION_SECONDS);
        System.out.println("total_requests=" + total);
        System.out.println("qps=" + qps);
        System.out.println("p50_ms=" + (p50 / 1e6));
        System.out.println("p99_ms=" + (p99 / 1e6));
        System.out.println("max_ms=" + (max / 1e6));
    }
}
