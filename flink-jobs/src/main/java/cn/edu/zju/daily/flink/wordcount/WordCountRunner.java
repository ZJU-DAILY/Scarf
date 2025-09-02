package cn.edu.zju.daily.flink.wordcount;

import cn.edu.zju.daily.flink.wordcount.source.TimestampedString;
import cn.edu.zju.daily.flink.wordcount.source.WordCountGeneratorConfig;
import cn.edu.zju.daily.flink.wordcount.source.WordCountSource;
import java.io.Serial;
import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.state.ReducingState;
import org.apache.flink.api.common.state.ReducingStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.v2.DiscardingSink;
import org.apache.flink.util.Collector;

public class WordCountRunner implements Runnable {

    private final WordCountGeneratorConfig config;

    private final int sourceParallelism;
    private final int mapParallelism;
    private final int reduceParallelism;
    private final int sinkParallelism;

    /**
     * Constructor for WordCountRunner.
     *
     * @param config WordCountGeneratorConfig
     * @param sourceParallelism source parallelism (0 if not set)
     * @param mapParallelism map parallelism (0 if not set)
     * @param reduceParallelism reduce parallelism (0 if not set)
     * @param sinkParallelism sink parallelism (0 if not set)
     */
    public WordCountRunner(
            WordCountGeneratorConfig config,
            int sourceParallelism,
            int mapParallelism,
            int reduceParallelism,
            int sinkParallelism) {
        this.config = config;
        this.sourceParallelism =
                sourceParallelism > 0 ? sourceParallelism : ExecutionConfig.PARALLELISM_DEFAULT;
        this.mapParallelism =
                mapParallelism > 0 ? mapParallelism : ExecutionConfig.PARALLELISM_DEFAULT;
        this.reduceParallelism =
                reduceParallelism > 0 ? reduceParallelism : ExecutionConfig.PARALLELISM_DEFAULT;
        this.sinkParallelism =
                sinkParallelism > 0 ? sinkParallelism : ExecutionConfig.PARALLELISM_DEFAULT;
    }

    public void run() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.disableOperatorChaining();

        WordCountSource source = new WordCountSource(config);

        DataStream<TimestampedString> text =
                env.fromSource(
                                source,
                                WatermarkStrategy.<TimestampedString>forMonotonousTimestamps()
                                        .withTimestampAssigner(
                                                (event, timestamp) -> event.getTimestamp()),
                                "source")
                        .uid("source")
                        .setParallelism(sourceParallelism);
        DataStream<Tuple2<String, Long>> counts =
                text.rebalance()
                        .flatMap(new Tokenizer())
                        .name("map")
                        .uid("map")
                        .setParallelism(mapParallelism)
                        .keyBy(value -> value.f0)
                        .flatMap(new CountWords())
                        .name("reduce")
                        .uid("reduce")
                        .setParallelism(reduceParallelism);
        counts.sinkTo(new DiscardingSink<>())
                .name("sink")
                .uid("sink")
                .setParallelism(sinkParallelism);

        try {
            env.executeAsync("WordCount");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static final class Tokenizer
            implements FlatMapFunction<TimestampedString, Tuple2<String, Long>> {
        @Serial private static final long serialVersionUID = 1L;

        @Override
        public void flatMap(TimestampedString value, Collector<Tuple2<String, Long>> out)
                throws Exception {
            // normalize and split the line
            String[] tokens = value.getValue().toLowerCase().split("\\W+");

            // emit the pairs
            for (String token : tokens) {
                if (!token.isEmpty()) {
                    out.collect(new Tuple2<>(token, 1L));
                }
            }
        }
    }

    public static final class CountWords
            extends RichFlatMapFunction<Tuple2<String, Long>, Tuple2<String, Long>> {

        private transient ReducingState<Long> count;

        @Override
        public void open(OpenContext openContext) throws Exception {

            ReducingStateDescriptor<Long> descriptor =
                    new ReducingStateDescriptor<Long>(
                            "count", // the state name
                            new Count(),
                            BasicTypeInfo.LONG_TYPE_INFO);

            count = getRuntimeContext().getReducingState(descriptor);
        }

        @Override
        public void flatMap(Tuple2<String, Long> value, Collector<Tuple2<String, Long>> out)
                throws Exception {
            count.add(value.f1);
            out.collect(new Tuple2<>(value.f0, count.get()));
        }

        public static final class Count implements ReduceFunction<Long> {

            @Override
            public Long reduce(Long value1, Long value2) throws Exception {
                return value1 + value2;
            }
        }
    }
}
