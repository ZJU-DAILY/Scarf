package cn.edu.zju.daily.flink.source;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import lombok.extern.slf4j.Slf4j;
import org.apache.flink.api.connector.source.*;
import org.apache.flink.core.io.SimpleVersionedSerializer;

@Slf4j
public abstract class ElasticSource<E, T>
        implements Source<T, ElasticSourceSplit, Collection<ElasticSourceSplit>> {

    public static final int NUM_SPLITS = 128;
    private final long maxEvents;
    private final long firstEventId;
    private final long stageDurationMillis;
    private final double[] localDelayUs;

    protected ElasticSource(
            long maxEvents, long firstEventId, long stageDurationMillis, double[] localDelayUs) {
        this.maxEvents = maxEvents;
        this.firstEventId = firstEventId;
        this.stageDurationMillis = stageDurationMillis;
        this.localDelayUs = localDelayUs;
    }

    @Override
    public Boundedness getBoundedness() {
        return Boundedness.CONTINUOUS_UNBOUNDED;
    }

    @Override
    public SplitEnumerator<ElasticSourceSplit, Collection<ElasticSourceSplit>> createEnumerator(
            SplitEnumeratorContext<ElasticSourceSplit> enumContext) throws Exception {
        LOG.info("Creating elastic enumerator");
        return new ElasticSplitEnumerator(enumContext, createSplits());
    }

    @Override
    public SplitEnumerator<ElasticSourceSplit, Collection<ElasticSourceSplit>> restoreEnumerator(
            SplitEnumeratorContext<ElasticSourceSplit> enumContext,
            Collection<ElasticSourceSplit> recoveredSplits)
            throws Exception {
        LOG.info("Restoring elastic enumerator");
        return new ElasticSplitEnumerator(enumContext, Collections.emptyList());
    }

    private Collection<ElasticSourceSplit> createSplits() {
        ArrayList<ElasticSourceSplit> splits = new ArrayList<>(NUM_SPLITS);

        long subMaxEvents = maxEvents / NUM_SPLITS;
        long subFirstEventId = firstEventId;

        for (int i = 0; i < NUM_SPLITS; i++) {
            if (i == NUM_SPLITS - 1) {
                // Don't lose any events to round-down.
                subMaxEvents = maxEvents - subMaxEvents * (NUM_SPLITS - 1);
            }
            splits.add(
                    new ElasticSourceSplit(
                            i,
                            NUM_SPLITS,
                            subFirstEventId,
                            subFirstEventId + subMaxEvents,
                            stageDurationMillis,
                            localDelayUs));
            subFirstEventId += subMaxEvents;
        }
        LOG.info("Created {} splits", splits.size());
        return splits;
    }

    @Override
    public SimpleVersionedSerializer<ElasticSourceSplit> getSplitSerializer() {
        return ElasticSourceSplitSerializer.INSTANCE;
    }

    @Override
    public SimpleVersionedSerializer<Collection<ElasticSourceSplit>>
            getEnumeratorCheckpointSerializer() {
        return ElasticSourceSplitCollectionSerializer.INSTANCE;
    }

    @Override
    public abstract SourceReader<T, ElasticSourceSplit> createReader(
            SourceReaderContext readerContext) throws Exception;

    public static class ElasticSourceSplitSerializer
            implements SimpleVersionedSerializer<ElasticSourceSplit> {

        private static final int VERSION = 1;
        public static ElasticSourceSplitSerializer INSTANCE = new ElasticSourceSplitSerializer();

        @Override
        public int getVersion() {
            return VERSION;
        }

        @Override
        public byte[] serialize(ElasticSourceSplit split) throws IOException {
            try (ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                    ObjectOutputStream objectOutputStream =
                            new ObjectOutputStream(byteArrayOutputStream)) {
                objectOutputStream.writeObject(split);
                return byteArrayOutputStream.toByteArray();
            }
        }

        @Override
        public ElasticSourceSplit deserialize(int version, byte[] serialized) throws IOException {
            if (version == VERSION) {
                try (ObjectInputStream objectInputStream =
                        new ObjectInputStream(new ByteArrayInputStream(serialized))) {
                    return (ElasticSourceSplit) objectInputStream.readObject();
                } catch (ClassNotFoundException e) {
                    throw new IOException("Failed to deserialize ElasticSourceSplit", e);
                }
            } else {
                throw new IOException("Unsupported version: " + version);
            }
        }
    }

    public static class ElasticSourceSplitCollectionSerializer
            implements SimpleVersionedSerializer<Collection<ElasticSourceSplit>> {

        private static final int VERSION = 1;
        public static ElasticSourceSplitCollectionSerializer INSTANCE =
                new ElasticSourceSplitCollectionSerializer();

        @Override
        public int getVersion() {
            return VERSION;
        }

        @Override
        public byte[] serialize(Collection<ElasticSourceSplit> splits) throws IOException {
            try (ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                    ObjectOutputStream objectOutputStream =
                            new ObjectOutputStream(byteArrayOutputStream)) {

                objectOutputStream.writeInt(splits.size());
                for (ElasticSourceSplit split : splits) {
                    objectOutputStream.writeObject(split);
                }
                return byteArrayOutputStream.toByteArray();
            }
        }

        @Override
        public Collection<ElasticSourceSplit> deserialize(int version, byte[] serialized)
                throws IOException {
            if (version == VERSION) {
                try (ObjectInputStream objectInputStream =
                        new ObjectInputStream(new ByteArrayInputStream(serialized))) {
                    int size = objectInputStream.readInt();
                    ArrayList<ElasticSourceSplit> splits = new ArrayList<>(size);
                    for (int i = 0; i < size; i++) {
                        splits.add((ElasticSourceSplit) objectInputStream.readObject());
                    }
                    return splits;
                } catch (ClassNotFoundException e) {
                    throw new IOException("Failed to deserialize ElasticSourceSplit", e);
                }
            } else {
                throw new IOException("Unsupported version: " + version);
            }
        }
    }
}
