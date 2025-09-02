package cn.edu.zju.daily.flink.source;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import lombok.extern.slf4j.Slf4j;
import org.apache.flink.connector.base.source.reader.RecordsBySplits;
import org.apache.flink.connector.base.source.reader.RecordsWithSplitIds;
import org.apache.flink.connector.base.source.reader.splitreader.SplitReader;
import org.apache.flink.connector.base.source.reader.splitreader.SplitsAddition;
import org.apache.flink.connector.base.source.reader.splitreader.SplitsChange;

@Slf4j
public class ElasticSplitReader<E> implements SplitReader<E, ElasticSourceSplit> {

    private static final int MAX_BATCH_SIZE = 1000;

    private final List<ElasticSourceSplit> splits;
    private final RawEventGenerator<E> generator;
    private final AtomicBoolean wakeup = new AtomicBoolean(false);
    private int currentIndex = 0;
    private final Object lock;
    // "now" perceived by the reader at the beginning. Semantically this is the timestamp when the
    // savepoint that this job recovers from is created.
    private final long now;

    public ElasticSplitReader(RawEventGenerator<E> generator, Object lock, long now) {
        this.splits = new ArrayList<>();
        this.lock = lock;
        this.generator = generator;
        this.now = now;
    }

    @Override
    public RecordsWithSplitIds<E> fetch() throws IOException {

        Map<String, Collection<E>> records = new HashMap<>();

        wakeup.compareAndSet(true, false);

        int count = 0;
        while (true) {
            if (wakeup.get()) {
                break;
            }

            if (count >= MAX_BATCH_SIZE) {
                break;
            }
            count++;

            ElasticSourceSplit split = splits.get(currentIndex);
            try {
                E event = generator.next(split);
                records.computeIfAbsent(split.splitId(), k -> new ArrayList<>()).add(event);
            } catch (InterruptedException e) {
                break;
            }

            currentIndex = (currentIndex + 1) % splits.size();
        }

        return new RecordsBySplits<>(records, Collections.emptySet());
    }

    @Override
    public void handleSplitsChanges(SplitsChange<ElasticSourceSplit> splitsChange) {
        // Get all the partition assignments and stopping offsets.
        if (!(splitsChange instanceof SplitsAddition)) {
            throw new UnsupportedOperationException(
                    String.format(
                            "The SplitChange type of %s is not supported.",
                            splitsChange.getClass()));
        }

        List<ElasticSourceSplit> splits = splitsChange.splits();
        long now = (this.now == 0) ? System.currentTimeMillis() : this.now;

        for (ElasticSourceSplit split : splits) {
            if (split.getOffsetEventId() <= 0) {
                // new split
                split.setBaseEventTime(now);
                split.setBaseEmitTime(now);
                split.setOffsetEventId(0);
                // baseEventId is set by ElasticSource.createSplits

                LOG.info(
                        "New split: baseEventTime = {}, baseEmitTime = {}, baseEventId = {}, offsetEventId = {}",
                        split.getBaseEventTime(),
                        split.getBaseEmitTime(),
                        split.getBaseEventId(),
                        split.getOffsetEventId());
            } else {
                // recovered split
                // baseEventTime is recovered
                split.setBaseEmitTime(now - split.getNextTimeOffsetMillis());
                // offsetEventId is recovered
                // baseEventId is recovered
                LOG.info(
                        "Recovered split: baseEventTime = {}, baseEmitTime = {}, baseEventId = {}, offsetEventId = {}",
                        split.getBaseEventTime(),
                        split.getBaseEmitTime(),
                        split.getBaseEventId(),
                        split.getOffsetEventId());
            }
        }

        LOG.info("Added {} splits to the split reader", splits.size());
        this.splits.addAll(splits);
    }

    @Override
    public void wakeUp() {
        wakeup.compareAndSet(false, true);
        synchronized (lock) {
            lock.notifyAll();
        }
    }

    @Override
    public void close() throws Exception {}
}
