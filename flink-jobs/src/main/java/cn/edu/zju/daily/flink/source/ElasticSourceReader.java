package cn.edu.zju.daily.flink.source;

import cn.edu.zju.daily.flink.utils.ValueGauge;
import java.util.Map;
import java.util.function.Supplier;
import org.apache.flink.api.connector.source.SourceReaderContext;
import org.apache.flink.connector.base.source.reader.RecordEmitter;
import org.apache.flink.connector.base.source.reader.SingleThreadMultiplexSourceReaderBase;
import org.apache.flink.connector.base.source.reader.splitreader.SplitReader;

/**
 * Source reader that supports parallelism adjustment across checkpoints.
 *
 * @param <E> Raw element type
 * @param <T> Emitted element type
 */
public class ElasticSourceReader<E, T>
        extends SingleThreadMultiplexSourceReaderBase<
                E, T, ElasticSourceSplit, ElasticSourceSplitState> {

    protected final ValueGauge<Long> watermarkDelayGauge;

    public ElasticSourceReader(
            Supplier<SplitReader<E, ElasticSourceSplit>> splitReaderSupplier,
            RecordEmitter<E, T, ElasticSourceSplitState> recordEmitter,
            SourceReaderContext context) {
        super(splitReaderSupplier, recordEmitter, context.getConfiguration(), context);
        watermarkDelayGauge = context.metricGroup().gauge("watermarkDelay", new ValueGauge<>(0L));
    }

    @Override
    public void start() {
        if (getNumberOfCurrentlyAssignedSplits() == 0) {
            context.sendSplitRequest();
        }
    }

    @Override
    protected void onSplitFinished(Map<String, ElasticSourceSplitState> finishedSplitIds) {
        throw new RuntimeException("Splits should not finish in Nexmark source");
    }

    @Override
    protected ElasticSourceSplitState initializedState(ElasticSourceSplit split) {
        return new ElasticSourceSplitState(split);
    }

    @Override
    protected ElasticSourceSplit toSplitType(String splitId, ElasticSourceSplitState splitState) {
        return splitState.getSplit();
    }
}
