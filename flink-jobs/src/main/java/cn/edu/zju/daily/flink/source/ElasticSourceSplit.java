package cn.edu.zju.daily.flink.source;

import java.io.Serializable;
import java.util.Arrays;
import lombok.Getter;
import lombok.Setter;
import org.apache.flink.api.connector.source.SourceSplit;

public final class ElasticSourceSplit implements SourceSplit, Serializable {

    private final String splitId;
    @Getter private final int id;
    @Getter private final int numSplits;
    @Getter @Setter private long baseEventId;
    private final long maxEventId;

    private final long stepNanos;
    private final long[] windowCapacities;
    private final long allWindowCapacity;
    @Getter private final long[] delayNanos; // in this split

    // States
    @Getter @Setter private long baseEmitTime;
    @Getter @Setter private long baseEventTime;
    @Getter @Setter private long offsetEventId;

    public ElasticSourceSplit(
            int id,
            int numSplits,
            long baseEventId,
            long maxEventId,
            long stageDurationMillis,
            double[] localDelayUs) {
        this.id = id;
        this.splitId = String.valueOf(id);
        this.numSplits = numSplits;
        this.baseEventId = baseEventId;
        this.maxEventId = maxEventId;
        this.offsetEventId = 0;
        this.stepNanos = stageDurationMillis * 1_000_000L;

        this.windowCapacities = new long[localDelayUs.length];
        this.delayNanos = new long[localDelayUs.length];
        for (int i = 0; i < localDelayUs.length; i++) {
            this.windowCapacities[i] = (long) (1_000L * stageDurationMillis / localDelayUs[i]);
            this.delayNanos[i] = (long) (1_000L * localDelayUs[i]);
        }
        this.allWindowCapacity = Arrays.stream(this.windowCapacities).sum();
    }

    @Override
    public String splitId() {
        return splitId;
    }

    public boolean hasNext() {
        return getNextEventId() < maxEventId;
    }

    public void incrementCount() {
        offsetEventId++;
    }

    public long getNextEventId() {
        return baseEventId + offsetEventId;
    }

    long getNextTimeOffsetMillis() {
        long allWindows = Math.floorDiv(offsetEventId, allWindowCapacity);
        long remainder = Math.floorMod(offsetEventId, allWindowCapacity);

        long nano = allWindows * stepNanos * windowCapacities.length;

        for (int i = 0; i < windowCapacities.length && remainder > 0; i++) {
            if (remainder < windowCapacities[i]) {
                nano += remainder * delayNanos[i];
                break;
            } else {
                nano += stepNanos;
                remainder -= windowCapacities[i];
            }
        }
        return nano / 1_000_000L;
    }

    public long getNextEventTimeMillis() {
        return baseEventTime + getNextTimeOffsetMillis();
    }

    public long getNextEmitTimeMillis() {
        return baseEmitTime + getNextTimeOffsetMillis();
    }

    @Override
    public String toString() {
        return "NexmarkSourceSplit{"
                + "id="
                + id
                + ", numSplits="
                + numSplits
                + ", baseEventId="
                + baseEventId
                + ", stepNanos="
                + stepNanos
                + ", windowCapacities="
                + Arrays.toString(windowCapacities)
                + ", allWindowCapacity="
                + allWindowCapacity
                + ", delayNanos="
                + Arrays.toString(delayNanos)
                + ", baseTimeMillis="
                + baseEmitTime
                + ", count="
                + offsetEventId
                + '}';
    }
}
