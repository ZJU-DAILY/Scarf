package cn.edu.zju.daily.flink.wordcount.source;

import java.io.Serializable;
import lombok.Getter;

@Getter
public class WordCountGeneratorConfig implements Serializable {

    private final int maxWordsPerLine;

    private final String resourceName;

    /**
     * Delay between events, in microseconds. If the array has more than one entry then the rate is
     * changed every {@link #rateIntervalSeconds}, and wraps around. This is the event delay of a
     * single split.
     */
    private final double[] localEventDelayUs;

    /** Delay before changing the current inter-event delay. */
    private final long rateIntervalSeconds;

    /** Time for first event (ms since epoch). */
    private final long baseTime;

    /**
     * Event id of first event to be generated. Event ids are unique over all generators, and are
     * used as a seed to generate each event's data.
     */
    public final long firstEventId;

    /** Maximum number of events to generate. */
    public final long maxEvents;

    public WordCountGeneratorConfig(
            int maxWordsPerLine,
            String resourceName,
            String eventRates,
            long rateIntervalSeconds,
            long baseTime,
            long firstEventId,
            long maxEvents) {
        this.maxWordsPerLine = maxWordsPerLine;
        this.resourceName = resourceName;

        String[] eventRatesArray = eventRates.split(",");
        this.localEventDelayUs = new double[eventRatesArray.length];
        for (int i = 0; i < eventRatesArray.length; i++) {
            this.localEventDelayUs[i] =
                    1_000_000d
                            / (Double.parseDouble(eventRatesArray[i]))
                            * WordCountSource.NUM_SPLITS;
        }

        this.rateIntervalSeconds = rateIntervalSeconds;
        this.baseTime = baseTime;
        this.firstEventId = firstEventId;
        this.maxEvents = maxEvents;
    }
}
