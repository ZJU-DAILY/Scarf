package cn.edu.zju.daily.flink.workload;

import static cn.edu.zju.daily.flink.yahoo.source.DataGenerator.*;

import cn.edu.zju.daily.flink.yahoo.AdvertisingTopologyNative;
import cn.edu.zju.daily.flink.yahoo.source.YahooConfig;
import java.util.Objects;
import org.apache.commons.cli.*;

public class Yahoo {
    public static void main(String[] args) throws ParseException {
        CommandLine cmd = new DefaultParser().parse(prepareOptions(), args);

        String sourceParallelism = cmd.getOptionValue("scp");
        String filterParallelism = cmd.getOptionValue("fp");
        String projectParallelism = cmd.getOptionValue("pp");
        String flatMapParallelism = cmd.getOptionValue("fmp");
        String aggregateParallelism = cmd.getOptionValue("ap");
        String sinkParallelism = cmd.getOptionValue("skp");

        String eventRates = cmd.getOptionValue("r");
        String rateInterval = cmd.getOptionValue("i");
        String baseTime = cmd.getOptionValue("b");

        YahooConfig config =
                new YahooConfig(
                        eventRates,
                        parseInt(rateInterval),
                        Long.parseLong(baseTime),
                        1,
                        Long.MAX_VALUE >>> 4,
                        makeIds(NUM_ADS),
                        makeIds(NUM_PAGES),
                        makeIds(NUM_USERS),
                        makeIds(NUM_CAMPAIGNS));

        AdvertisingTopologyNative runner =
                new AdvertisingTopologyNative(
                        config,
                        parseInt(sourceParallelism),
                        parseInt(filterParallelism),
                        parseInt(projectParallelism),
                        parseInt(flatMapParallelism),
                        parseInt(aggregateParallelism),
                        parseInt(sinkParallelism));
        runner.run();
    }

    private static int parseInt(String str) {
        return Objects.isNull(str) ? 0 : Integer.parseInt(str);
    }

    private static Options prepareOptions() {
        Option eventRatesOption =
                Option.builder("r")
                        .longOpt("event-rates")
                        .required(true)
                        .hasArg()
                        .desc("Events per second, separated by commas")
                        .build();
        Option rateIntervalSeconds =
                Option.builder("i")
                        .longOpt("rate-interval-seconds")
                        .hasArg()
                        .desc("The rate interval in seconds")
                        .build();

        // Parallelisms
        Option sourceParallelismOption =
                Option.builder("scp")
                        .longOpt("source.parallelism")
                        .hasArg()
                        .desc("The parallelism of the source")
                        .build();

        Option filterParallelismOption =
                Option.builder("fp")
                        .longOpt("filter.parallelism")
                        .hasArg()
                        .desc("The parallelism of the filter operator")
                        .build();

        Option projectParallelismOption =
                Option.builder("pp")
                        .longOpt("project.parallelism")
                        .hasArg()
                        .desc("The parallelism of the project operator")
                        .build();

        Option flatMapParallelismOption =
                Option.builder("fmp")
                        .longOpt("flatmap.parallelism")
                        .hasArg()
                        .desc("The parallelism of the flatMap operator")
                        .build();

        Option aggregateParallelismOption =
                Option.builder("ap")
                        .longOpt("aggregate.parallelism")
                        .hasArg()
                        .desc("The parallelism of the aggregate operator")
                        .build();

        Option sinkParallelismOption =
                Option.builder("skp")
                        .longOpt("sink.parallelism")
                        .hasArg()
                        .desc("The parallelism of the sink operator")
                        .build();

        Option baseTimeOption =
                Option.builder("b")
                        .longOpt("base-time")
                        .hasArg()
                        .desc(
                                "Timestamp of first event in milliseconds since epoch. Used only when not recovering from a checkpoint.")
                        .build();

        return new Options()
                .addOption(eventRatesOption)
                .addOption(rateIntervalSeconds)
                .addOption(baseTimeOption)
                .addOption(sourceParallelismOption)
                .addOption(filterParallelismOption)
                .addOption(projectParallelismOption)
                .addOption(flatMapParallelismOption)
                .addOption(aggregateParallelismOption)
                .addOption(sinkParallelismOption);
    }
}
