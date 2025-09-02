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

package cn.edu.zju.daily.flink.nexmark.source;

import cn.edu.zju.daily.flink.nexmark.generator.NexmarkGenerator;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import lombok.extern.slf4j.Slf4j;
import org.apache.flink.api.connector.source.ReaderOutput;
import org.apache.flink.api.connector.source.SourceReader;
import org.apache.flink.api.connector.source.SourceReaderContext;
import org.apache.flink.core.io.InputStatus;
import org.apache.flink.metrics.Counter;
import org.apache.flink.table.data.RowData;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.concurrent.ExecutorThreadFactory;

/**
 * @deprecated Use {@link cn.edu.zju.daily.flink.nexmark.source.v2.NexmarkSourceReader} instead.
 */
@Slf4j
@Deprecated
public class NexmarkSourceReader
        implements SourceReader<RowData, NexmarkSource.NexmarkSourceSplit> {

    private static final long BATCH_INTERVAL = 500;

    private final SourceReaderContext context;
    private final EventDeserializer<RowData> deserializer;
    private final Counter numRecordsInCounter;
    private NexmarkSource.NexmarkSourceSplit sourceSplit;
    private NexmarkGenerator generator;
    private final Executor limiter =
            Executors.newSingleThreadExecutor(new ExecutorThreadFactory("source-reader-limiter"));

    private long nextEventWallclockTimeStamp = 0;

    NexmarkSourceReader(
            SourceReaderContext sourceReaderContext, EventDeserializer<RowData> deserializer) {
        this.context = sourceReaderContext;
        this.deserializer = deserializer;
        this.numRecordsInCounter =
                context.metricGroup().getIOMetricGroup().getNumRecordsInCounter();
    }

    @Override
    public void start() {
        context.sendSplitRequest();
    }

    @Override
    public InputStatus pollNext(ReaderOutput<RowData> readerOutput) throws Exception {
        if (sourceSplit == null || generator == null) {
            return InputStatus.NOTHING_AVAILABLE;
        }
        if (!generator.hasNext()) {
            return InputStatus.END_OF_INPUT;
        }
        long now = System.currentTimeMillis();

        if (generator.getWallclockBaseTime() < 0) {
            generator.setWallclockBaseTime(now);
            sourceSplit.setWallclockBaseTime(now);
        }
        NexmarkGenerator.NextEvent nextEvent = generator.next();
        if (nextEvent.wallclockTimestamp - now > BATCH_INTERVAL) {
            nextEventWallclockTimeStamp = nextEvent.wallclockTimestamp;
            return InputStatus.NOTHING_AVAILABLE;
        }
        readerOutput.collect(deserializer.deserialize(nextEvent.event));
        numRecordsInCounter.inc();
        sourceSplit.setNumEmittedSoFar(generator.getEventsCountSoFar());
        return InputStatus.MORE_AVAILABLE;
    }

    @Override
    public List<NexmarkSource.NexmarkSourceSplit> snapshotState(long l) {
        return Collections.singletonList(sourceSplit);
    }

    @Override
    public CompletableFuture<Void> isAvailable() {
        long now = System.currentTimeMillis();
        if (nextEventWallclockTimeStamp <= now) {
            return CompletableFuture.completedFuture(null);
        }
        return CompletableFuture.runAsync(
                () -> {
                    try {
                        Thread.sleep(nextEventWallclockTimeStamp - now);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                },
                limiter);
    }

    @Override
    public void addSplits(List<NexmarkSource.NexmarkSourceSplit> list) {
        Preconditions.checkState(list.size() == 1, "Only one split supported for one reader");
        Preconditions.checkState(sourceSplit == null, "We already have one split.");
        sourceSplit = list.get(0);

        sourceSplit.setNumEmittedSoFar(0);
        sourceSplit.setWallclockBaseTime(-1);

        generator =
                new NexmarkGenerator(
                        sourceSplit.getGeneratorConfig(),
                        sourceSplit.getNumEmittedSoFar(),
                        sourceSplit.getWallclockBaseTime());

        LOG.info(
                "Added new source split with config: {}, wallclockBaseTime: {}, numEmittedSoFar: {}",
                generator.getCurrentConfig(),
                generator.getWallclockBaseTime(),
                sourceSplit.getNumEmittedSoFar());
    }

    @Override
    public void notifyNoMoreSplits() {}

    @Override
    public void close() throws Exception {}
}
