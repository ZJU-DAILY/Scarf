package cn.edu.zju.daily.flink.workload;

import cn.edu.zju.daily.flink.nexmark.QueryRunner;
import cn.edu.zju.daily.flink.nexmark.workload.Workload;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

public class Nexmark {

    public static void main(String[] args) throws Exception {

        CommandLine cmd = new DefaultParser().parse(prepareOptions(), args);
        String queryName = cmd.getOptionValue("query-name");
        String eventRates = cmd.getOptionValue("event-rates");
        String rateInterval = cmd.getOptionValue("rate-interval-seconds");
        String kafkaServers = cmd.getOptionValue("kafka-servers");
        String kafkaHome = cmd.getOptionValue("kafka-home");
        boolean explainOnly = cmd.hasOption("explain-only");
        String planSaveFileName = cmd.getOptionValue("plan-save-file-name");
        long baseTime = Long.parseLong(cmd.getOptionValue("base-time", "0"));

        // Both of none of Kafka server and kafka home should be null
        if (kafkaServers == null ^ kafkaHome == null) {
            throw new IllegalArgumentException(
                    "Both or none of Kafka server and kafka home should be set");
        }

        Workload workload =
                new Workload(eventRates, Integer.parseInt(rateInterval), 0, 1, 3, 46, kafkaServers);
        QueryRunner runner =
                new QueryRunner(
                        queryName, workload, kafkaHome, explainOnly, planSaveFileName, baseTime);
        runner.run();
    }

    private static Options prepareOptions() {
        Option queryNameOption =
                Option.builder("q")
                        .longOpt("query-name")
                        .required()
                        .hasArg()
                        .desc("The name of the query")
                        .build();
        Option eventRatesOption =
                Option.builder("r")
                        .longOpt("event-rates")
                        .required()
                        .hasArg()
                        .desc("The event rates (tps)")
                        .build();
        Option rateIntervalSecondsOption =
                Option.builder("i")
                        .longOpt("rate-interval-seconds")
                        .hasArg()
                        .desc("The rate interval in seconds")
                        .build();
        Option kafkaServersOption =
                Option.builder("s")
                        .longOpt("kafka-servers")
                        .hasArg()
                        .desc("The kafka servers")
                        .build();
        Option kafkaPathOption =
                Option.builder("h")
                        .longOpt("kafka-home")
                        .hasArg()
                        .desc("The kafka home path")
                        .build();
        Option getPlanOption =
                Option.builder("e")
                        .longOpt("explain-only")
                        .desc("Get the plan of the query")
                        .build();
        Option planSaveFileNameOption =
                Option.builder("p")
                        .longOpt("plan-save-file-name")
                        .hasArg()
                        .desc("The path to save the plan")
                        .build();
        Option baseTimeOption =
                Option.builder("b")
                        .longOpt("base-time")
                        .hasArg()
                        .desc(
                                "Timestamp of first event in milliseconds since epoch. Used only when not recovering from a checkpoint.")
                        .build();
        Options options = new Options();
        options.addOption(queryNameOption);
        options.addOption(eventRatesOption);
        options.addOption(rateIntervalSecondsOption);
        options.addOption(kafkaServersOption);
        options.addOption(kafkaPathOption);
        options.addOption(getPlanOption);
        options.addOption(planSaveFileNameOption);
        options.addOption(baseTimeOption);

        return options;
    }
}
