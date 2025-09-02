package cn.edu.zju.daily.flink.workload;

import cn.edu.zju.daily.flink.wordcount.WordCountRunner;
import cn.edu.zju.daily.flink.wordcount.source.WordCountGeneratorConfig;
import java.util.Objects;
import org.apache.commons.cli.*;

public class WordCount {

    public static void main(String[] args) throws ParseException {
        CommandLine cmd = new DefaultParser().parse(prepareOptions(), args);

        String sourceParallelism = cmd.getOptionValue("scp");
        String mapParallelism = cmd.getOptionValue("mp");
        String reduceParallelism = cmd.getOptionValue("rp");
        String sinkParallelism = cmd.getOptionValue("skp");

        String maxWordsPerLine = cmd.getOptionValue("l");
        String resourceName = cmd.getOptionValue("t");
        String eventRates = cmd.getOptionValue("r");
        String rateInterval = cmd.getOptionValue("i");
        String baseTime = cmd.getOptionValue("b");

        WordCountGeneratorConfig config =
                new WordCountGeneratorConfig(
                        parseInt(maxWordsPerLine),
                        resourceName,
                        eventRates,
                        parseInt(rateInterval),
                        Long.parseLong(baseTime),
                        1,
                        Long.MAX_VALUE >>> 4);

        WordCountRunner runner =
                new WordCountRunner(
                        config,
                        parseInt(sourceParallelism),
                        parseInt(mapParallelism),
                        parseInt(reduceParallelism),
                        parseInt(sinkParallelism));
        runner.run();
    }

    private static int parseInt(String str) {
        return Objects.isNull(str) ? 0 : Integer.parseInt(str);
    }

    private static Options prepareOptions() {
        Option resourceNameOption =
                Option.builder("t")
                        .longOpt("resource-name")
                        .required(true)
                        .hasArg()
                        .desc("The name of the resource")
                        .build();
        Option eventRatesOption =
                Option.builder("r")
                        .longOpt("event-rates")
                        .required(true)
                        .hasArg()
                        .desc("Events per second, separated by commas")
                        .build();
        Option maxWordsPerLineOption =
                Option.builder("l")
                        .longOpt("max-words-per-line")
                        .required(true)
                        .hasArg()
                        .desc("The maximum number of words per line")
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

        Option mapParallelismOption =
                Option.builder("mp")
                        .longOpt("map.parallelism")
                        .hasArg()
                        .desc("The parallelism of the map")
                        .build();

        Option reduceParallelismOption =
                Option.builder("rp")
                        .longOpt("reduce.parallelism")
                        .hasArg()
                        .desc("The parallelism of the reduce")
                        .build();

        Option sinkParallelismOption =
                Option.builder("skp")
                        .longOpt("sink.parallelism")
                        .hasArg()
                        .desc("The parallelism of the sink")
                        .build();

        Option baseTimeOption =
                Option.builder("b")
                        .longOpt("base-time")
                        .hasArg()
                        .desc(
                                "Timestamp of first event in milliseconds since epoch. Used only when not recovering from a checkpoint.")
                        .build();

        return new Options()
                .addOption(resourceNameOption)
                .addOption(eventRatesOption)
                .addOption(maxWordsPerLineOption)
                .addOption(rateIntervalSeconds)
                .addOption(baseTimeOption)
                .addOption(sourceParallelismOption)
                .addOption(mapParallelismOption)
                .addOption(reduceParallelismOption)
                .addOption(sinkParallelismOption);
    }
}
