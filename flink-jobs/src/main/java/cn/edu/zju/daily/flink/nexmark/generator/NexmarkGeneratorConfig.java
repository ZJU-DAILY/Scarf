/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cn.edu.zju.daily.flink.nexmark.generator;

import cn.edu.zju.daily.flink.nexmark.NexmarkConfiguration;
import cn.edu.zju.daily.flink.nexmark.model.Event;
import cn.edu.zju.daily.flink.nexmark.source.v2.NexmarkSource;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import lombok.Getter;

/** Parameters controlling how {@link NexmarkGenerator} synthesizes {@link Event} elements. */
public class NexmarkGeneratorConfig implements Serializable {

    /**
     * We start the ids at specific values to help ensure the queries find a match even on small
     * synthesized dataset sizes.
     */
    public static final long FIRST_AUCTION_ID = 1000L;

    public static final long FIRST_PERSON_ID = 1000L;
    public static final long FIRST_CATEGORY_ID = 10L;

    /** Proportions of people/auctions/bids to synthesize. */
    public final int personProportion;

    public final int auctionProportion;
    public final int bidProportion;
    public final int totalProportion;

    /** Environment options. */
    private final NexmarkConfiguration configuration;

    /**
     * Delay between events, in microseconds. If the array has more than one entry then the rate is
     * changed every {@link #stepLengthSec}, and wraps around. This is the event delay of a single
     * split.
     */
    @Getter private final double[] localEventDelayUs;

    /** Delay before changing the current inter-event delay. */
    @Getter private final long stepLengthSec;

    /** Time for first event (ms since epoch). */
    @Getter private final long baseTime;

    /**
     * Event id of first event to be generated. Event ids are unique over all generators, and are
     * used as a seed to generate each event's data.
     */
    public final long firstEventId;

    /** Maximum number of events to generate. */
    public final long maxEvents;

    /**
     * First event number. Generators running in parallel time may share the same event number, and
     * the event number is used to determine the event timestamp.
     */
    public final long firstEventNumber;

    /**
     * True period of epoch in milliseconds. Derived from above. (Ie time to run through cycle for
     * all interEventDelayUs entries).
     */
    private final long epochPeriodMs;

    /**
     * Number of events per epoch. Derived from above. (Ie number of events to run through cycle for
     * all interEventDelayUs entries).
     */
    private final long eventsPerEpoch;

    public NexmarkGeneratorConfig(
            NexmarkConfiguration configuration,
            long baseTime,
            long firstEventId,
            long maxEventsOrZero,
            long firstEventNumber) {

        this.auctionProportion = configuration.auctionProportion;
        this.personProportion = configuration.personProportion;
        this.bidProportion = configuration.bidProportion;
        this.totalProportion = this.auctionProportion + this.personProportion + this.bidProportion;

        this.configuration = configuration;

        String[] eventRatesArray = configuration.eventRates.split(",");
        this.localEventDelayUs = new double[eventRatesArray.length];
        for (int i = 0; i < eventRatesArray.length; i++) {
            this.localEventDelayUs[i] =
                    1_000_000d
                            / (Double.parseDouble(eventRatesArray[i]))
                            * NexmarkSource.NUM_SPLITS;
        }
        this.stepLengthSec = configuration.rateIntervalSec;
        this.baseTime = baseTime;
        this.firstEventId = firstEventId;

        if (maxEventsOrZero == 0) {
            // Scale maximum down to avoid overflow in getEstimatedSizeBytes.
            this.maxEvents =
                    Long.MAX_VALUE
                            / (totalProportion
                                    * Math.max(
                                            Math.max(
                                                    configuration.avgPersonByteSize,
                                                    configuration.avgAuctionByteSize),
                                            configuration.avgBidByteSize));
        } else {
            this.maxEvents = maxEventsOrZero;
        }
        this.firstEventNumber = firstEventNumber;

        long eventsPerEpoch = 0;
        long epochPeriodMs = 0;
        this.eventsPerEpoch = eventsPerEpoch;
        this.epochPeriodMs = epochPeriodMs;
    }

    /** Return a copy of this config. */
    public NexmarkGeneratorConfig copy() {
        NexmarkGeneratorConfig result;
        result =
                new NexmarkGeneratorConfig(
                        configuration, baseTime, firstEventId, maxEvents, firstEventNumber);
        return result;
    }

    /**
     * Split this config into {@code n} sub-configs with roughly equal number of possible events,
     * but distinct value spaces. The generators will run on parallel timelines. This config should
     * no longer be used.
     */
    public List<NexmarkGeneratorConfig> split(int n) {
        List<NexmarkGeneratorConfig> results = new ArrayList<>();
        if (n == 1) {
            // No split required.
            results.add(this);
        } else {
            long subMaxEvents = maxEvents / n;
            long subFirstEventId = firstEventId;
            for (int i = 0; i < n; i++) {
                if (i == n - 1) {
                    // Don't loose any events to round-down.
                    subMaxEvents = maxEvents - subMaxEvents * (n - 1);
                }
                results.add(copyWith(subFirstEventId, subMaxEvents, firstEventNumber));
                subFirstEventId += subMaxEvents;
            }
        }
        return results;
    }

    /** Return copy of this config except with given parameters. */
    public NexmarkGeneratorConfig copyWith(
            long firstEventId, long maxEvents, long firstEventNumber) {
        return new NexmarkGeneratorConfig(
                configuration, baseTime, firstEventId, maxEvents, firstEventNumber);
    }

    /** Return an estimate of the bytes needed by {@code numEvents}. */
    public long estimatedBytesForEvents(long numEvents) {
        long numPersons = (numEvents * personProportion) / totalProportion;
        long numAuctions = (numEvents * auctionProportion) / totalProportion;
        long numBids = (numEvents * bidProportion) / totalProportion;
        return numPersons * configuration.avgPersonByteSize
                + numAuctions * configuration.avgAuctionByteSize
                + numBids * configuration.avgBidByteSize;
    }

    public int getAvgPersonByteSize() {
        return configuration.avgPersonByteSize;
    }

    public int getNumActivePeople() {
        return configuration.numActivePeople;
    }

    public int getHotSellersRatio() {
        return configuration.hotSellersRatio;
    }

    public int getNumInFlightAuctions() {
        return configuration.numInFlightAuctions;
    }

    public int getHotAuctionRatio() {
        return configuration.hotAuctionRatio;
    }

    public int getHotBiddersRatio() {
        return configuration.hotBiddersRatio;
    }

    public int getAvgBidByteSize() {
        return configuration.avgBidByteSize;
    }

    public int getAvgAuctionByteSize() {
        return configuration.avgAuctionByteSize;
    }

    public double getProbDelayedEvent() {
        return configuration.probDelayedEvent;
    }

    public long getOccasionalDelaySec() {
        return configuration.occasionalDelaySec;
    }

    /**
     * Return an estimate of the byte-size of all events a generator for this config would yield.
     */
    public long getEstimatedSizeBytes() {
        return estimatedBytesForEvents(maxEvents);
    }

    /**
     * Return the first 'event id' which could be generated from this config. Though events don't
     * have ids we can simulate them to help bookkeeping.
     */
    public long getStartEventId() {
        return firstEventId + firstEventNumber;
    }

    /** Return one past the last 'event id' which could be generated from this config. */
    public long getStopEventId() {
        return firstEventId + firstEventNumber + maxEvents;
    }

    /** Return the next event number for a generator which has so far emitted {@code numEvents}. */
    public long nextEventNumber(long numEvents) {
        return firstEventNumber + numEvents;
    }

    /**
     * Return the next event number for a generator which has so far emitted {@code numEvents}, but
     * adjusted to account for {@code outOfOrderGroupSize}.
     */
    public long nextAdjustedEventNumber(long numEvents) {
        long n = configuration.outOfOrderGroupSize;
        long eventNumber = nextEventNumber(numEvents);
        long base = (eventNumber / n) * n;
        long offset = (eventNumber * 953) % n;
        return base + offset;
    }

    /**
     * Return the event number who's event time will be a suitable watermark for a generator which
     * has so far emitted {@code numEvents}.
     */
    public long nextEventNumberForWatermark(long numEvents) {
        long n = configuration.outOfOrderGroupSize;
        long eventNumber = nextEventNumber(numEvents);
        return (eventNumber / n) * n;
    }

    /** What timestamp should the event with {@code eventNumber} have for this generator? */
    public long timestampForEvent(long eventNumber) {
        return baseTime + (long) (eventNumber * localEventDelayUs[0]) / 1000L;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        NexmarkGeneratorConfig that = (NexmarkGeneratorConfig) o;
        return personProportion == that.personProportion
                && auctionProportion == that.auctionProportion
                && bidProportion == that.bidProportion
                && totalProportion == that.totalProportion
                && stepLengthSec == that.stepLengthSec
                && firstEventId == that.firstEventId
                && maxEvents == that.maxEvents
                && firstEventNumber == that.firstEventNumber
                && epochPeriodMs == that.epochPeriodMs
                && eventsPerEpoch == that.eventsPerEpoch
                && Objects.equals(configuration, that.configuration)
                && Arrays.equals(localEventDelayUs, that.localEventDelayUs);
    }

    @Override
    public int hashCode() {
        int result =
                Objects.hash(
                        personProportion,
                        auctionProportion,
                        bidProportion,
                        totalProportion,
                        configuration,
                        stepLengthSec,
                        firstEventId,
                        maxEvents,
                        firstEventNumber,
                        epochPeriodMs,
                        eventsPerEpoch);
        result = 31 * result + Arrays.hashCode(localEventDelayUs);
        return result;
    }

    @Override
    public String toString() {
        return "GeneratorConfig{"
                + "personProportion="
                + personProportion
                + ", auctionProportion="
                + auctionProportion
                + ", bidProportion="
                + bidProportion
                + ", totalProportion="
                + totalProportion
                + ", configuration="
                + configuration
                + ", interEventDelayUs="
                + Arrays.toString(localEventDelayUs)
                + ", stepLengthSec="
                + stepLengthSec
                + ", baseTime="
                + baseTime
                + ", firstEventId="
                + firstEventId
                + ", maxEvents="
                + maxEvents
                + ", firstEventNumber="
                + firstEventNumber
                + ", epochPeriodMs="
                + epochPeriodMs
                + ", eventsPerEpoch="
                + eventsPerEpoch
                + '}';
    }
}
