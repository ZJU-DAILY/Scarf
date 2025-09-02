package cn.edu.zju.daily.flink.source;

import java.io.IOException;
import java.util.*;
import javax.annotation.Nullable;
import lombok.extern.slf4j.Slf4j;
import org.apache.flink.api.connector.source.SplitEnumerator;
import org.apache.flink.api.connector.source.SplitEnumeratorContext;
import org.apache.flink.api.connector.source.SplitsAssignment;

@Slf4j
public class ElasticSplitEnumerator
        implements SplitEnumerator<ElasticSourceSplit, Collection<ElasticSourceSplit>> {

    private final SplitEnumeratorContext<ElasticSourceSplit> context;
    private final List<ElasticSourceSplit> splits;

    public ElasticSplitEnumerator(
            SplitEnumeratorContext<ElasticSourceSplit> context,
            Collection<ElasticSourceSplit> splits) {
        this.context = context;
        this.splits = new ArrayList<>(splits);
    }

    @Override
    public void start() {}

    @Override
    public void handleSplitRequest(int subtaskId, @Nullable String requesterHostname) {

        if (splits.isEmpty()) {
            LOG.info("No splits available for subtask {}. Sending empty assignment.", subtaskId);
            context.signalNoMoreSplits(subtaskId);
            return;
        }

        int parallelism = context.currentParallelism();
        int numSplits = splits.size();
        int numSplitsPerTask = numSplits / parallelism;

        int startId = 0;
        for (int i = 0; i < parallelism; i++) {
            int endId = startId + numSplitsPerTask;
            if (i < numSplits % parallelism) {
                endId++;
            }

            if (endId > numSplits) {
                throw new AssertionError("endId > numSplits");
            }

            if (i == subtaskId) {
                List<ElasticSourceSplit> subList = splits.subList(startId, endId);
                LOG.info("Assigning splits [{}, {}) to subtask {}: {}", startId, endId, i, subList);
                context.assignSplits(new SplitsAssignment<>(Collections.singletonMap(i, subList)));
                context.signalNoMoreSplits(i);
                return;
            }

            startId = endId;
        }
        throw new AssertionError("subtaskId >= context.currentParallelism()");
    }

    @Override
    public void addSplitsBack(List<ElasticSourceSplit> splits, int subtaskId) {
        splits.addAll(this.splits);
    }

    @Override
    public void addReader(int subtaskId) {}

    @Override
    public Collection<ElasticSourceSplit> snapshotState(long checkpointId) throws Exception {
        return new ArrayList<>(splits);
    }

    @Override
    public void close() throws IOException {}
}
