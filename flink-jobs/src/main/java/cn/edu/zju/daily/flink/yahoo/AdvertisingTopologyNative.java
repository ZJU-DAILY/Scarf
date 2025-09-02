/**
 * Copyright 2015, Yahoo Inc. Licensed under the terms of the Apache License 2.0. Please see LICENSE
 * file in the project root for terms.
 */
package cn.edu.zju.daily.flink.yahoo;

import static cn.edu.zju.daily.flink.yahoo.source.DataGenerator.ADS_PER_CAMPAIGN;

import cn.edu.zju.daily.flink.yahoo.advertising.Window;
import cn.edu.zju.daily.flink.yahoo.source.YahooConfig;
import cn.edu.zju.daily.flink.yahoo.source.YahooSource;
import java.time.Duration;
import java.util.*;
import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.*;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple7;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.v2.DiscardingSink;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

/**
 * To Run: flink run target/flink-benchmarks-0.1.0-AdvertisingTopologyNative.jar --confPath
 * "../conf/benchmarkConf.yaml"
 */
public class AdvertisingTopologyNative {

    private final YahooConfig config;
    private final int sourceParallelism;
    private final int filterParallelism;
    private final int projectParallelism;
    private final int flatMapParallelism;
    private final int aggregateParallelism;
    private final int sinkParallelism;

    public AdvertisingTopologyNative(
            YahooConfig config,
            int sourceParallelism,
            int filterParallelism,
            int projectParallelism,
            int flatMapParallelism,
            int aggregateParallelism,
            int sinkParallelism) {
        this.config = config;
        this.sourceParallelism =
                sourceParallelism > 0 ? sourceParallelism : ExecutionConfig.PARALLELISM_DEFAULT;
        this.filterParallelism =
                filterParallelism > 0 ? filterParallelism : ExecutionConfig.PARALLELISM_DEFAULT;
        this.projectParallelism =
                projectParallelism > 0 ? projectParallelism : ExecutionConfig.PARALLELISM_DEFAULT;
        this.flatMapParallelism =
                flatMapParallelism > 0 ? flatMapParallelism : ExecutionConfig.PARALLELISM_DEFAULT;
        this.aggregateParallelism =
                aggregateParallelism > 0
                        ? aggregateParallelism
                        : ExecutionConfig.PARALLELISM_DEFAULT;
        this.sinkParallelism =
                sinkParallelism > 0 ? sinkParallelism : ExecutionConfig.PARALLELISM_DEFAULT;
    }

    public void run() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.disableOperatorChaining();

        YahooSource source = new YahooSource(config);

        // userId, pageId, ads, adType, eventType, eventTime, IP
        DataStream<Tuple7<String, String, String, String, String, Long, String>> messageStream =
                env.fromSource(
                                source,
                                WatermarkStrategy
                                        .<Tuple7<
                                                        String,
                                                        String,
                                                        String,
                                                        String,
                                                        String,
                                                        Long,
                                                        String>>
                                                forMonotonousTimestamps()
                                        .withTimestampAssigner(
                                                (event, timestamp) -> event.getField(5)),
                                "source")
                        .setParallelism(sourceParallelism)
                        .uid("source");

        messageStream
                // Filter the records if event type is "view"
                .filter(new EventFilterBolt())
                .setParallelism(filterParallelism)
                .name("filter")
                .uid("filter")

                // project the event: ad, eventTime
                .<Tuple2<String, Long>>project(2, 5)
                .setParallelism(projectParallelism)
                .name("project")
                .uid("project")

                // perform join with redis data: campaignId, ad, eventTime
                .flatMap(new RedisJoinBolt(config))
                .setParallelism(flatMapParallelism)
                .name("flatmap")
                .uid("flatmap")

                // process campaign
                .<String>keyBy(tuple -> tuple.getField(0)) // key by campaign_id
                .window(TumblingEventTimeWindows.of(Duration.ofSeconds(1)))
                .<Integer, Integer, Window>aggregate(
                        new CampaignProcessor(), new CampaignWindowFunction())
                .setParallelism(aggregateParallelism)
                .name("aggregate")
                .uid("aggregate")

                // sink
                .sinkTo(new DiscardingSink<>())
                .setParallelism(sinkParallelism)
                .name("sink")
                .uid("sink");

        try {
            env.execute("Yahoo");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static class EventFilterBolt
            implements FilterFunction<
                    Tuple7<String, String, String, String, String, Long, String>> {
        @Override
        public boolean filter(Tuple7<String, String, String, String, String, Long, String> tuple)
                throws Exception {
            return tuple.getField(4).equals("view");
        }
    }

    private static final class RedisJoinBolt
            extends RichFlatMapFunction<Tuple2<String, Long>, Tuple3<String, String, Long>> {

        private final YahooConfig config;
        private Map<String, String> adToCampaign;

        RedisJoinBolt(YahooConfig config) {
            this.config = config;
        }

        @Override
        public void open(OpenContext context) {
            adToCampaign = new HashMap<>();
            List<String> campaigns = config.getCampaignIds();
            List<String> ads = config.getAdIds();
            for (int i = 0; i < campaigns.size(); i++) {
                String campaign = campaigns.get(i);
                List<String> campaignAds =
                        ads.subList(i * ADS_PER_CAMPAIGN, (i + 1) * ADS_PER_CAMPAIGN);
                for (String ad : campaignAds) {
                    adToCampaign.put(ad, campaign);
                }
            }
        }

        @Override
        public void flatMap(Tuple2<String, Long> input, Collector<Tuple3<String, String, Long>> out)
                throws Exception {
            String adId = input.getField(0);
            String campaignId = adToCampaign.get(adId);
            if (campaignId == null) {
                return;
            }

            Tuple3<String, String, Long> tuple =
                    new Tuple3<>(campaignId, input.getField(0), input.getField(1));
            out.collect(tuple);
        }
    }

    private static class CampaignProcessor
            implements AggregateFunction<Tuple3<String, String, Long>, Integer, Integer> {

        @Override
        public Integer createAccumulator() {
            return 0;
        }

        @Override
        public Integer add(Tuple3<String, String, Long> value, Integer accumulator) {
            return accumulator + 1;
        }

        @Override
        public Integer getResult(Integer accumulator) {
            return accumulator;
        }

        @Override
        public Integer merge(Integer a, Integer b) {
            return a + b;
        }
    }

    private static class CampaignWindowFunction
            implements WindowFunction<Integer, Window, String, TimeWindow> {

        @Override
        public void apply(
                String s, TimeWindow window, Iterable<Integer> input, Collector<Window> out)
                throws Exception {
            Window win = new Window(Long.toString(window.getStart()));
            for (Integer el : input) {
                win.seenCount += el;
            }
            out.collect(win);
        }
    }
}
