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

package cn.edu.zju.daily.flink.nexmark;

import cn.edu.zju.daily.flink.nexmark.utils.AutoClosableProcess;
import cn.edu.zju.daily.flink.nexmark.utils.HDFSUtils;
import cn.edu.zju.daily.flink.nexmark.workload.Workload;
import java.io.*;
import java.time.LocalDateTime;
import java.util.*;
import javax.annotation.Nullable;
import lombok.extern.slf4j.Slf4j;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.ExplainDetail;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

@Slf4j
public class QueryRunner implements Runnable {

    private final String queryName;
    private final Workload workload;
    private final String kafkaHome;
    private final boolean explainOnly;
    private final String planSaveFileName; // hdfs
    private final long baseTime;

    public QueryRunner(
            String queryName,
            Workload workload,
            @Nullable String kafkaHome,
            boolean explainOnly,
            String planSaveFileName,
            long baseTime) {
        this.queryName = queryName;
        this.workload = workload;
        if (kafkaHome != null) {
            kafkaHome = kafkaHome.strip();
            while (kafkaHome.endsWith("/")) {
                kafkaHome = kafkaHome.substring(0, kafkaHome.length() - 1);
            }
        }
        this.kafkaHome = kafkaHome;
        this.explainOnly = explainOnly;
        this.planSaveFileName = planSaveFileName;
        this.baseTime = baseTime;
    }

    public void run() {
        try {
            System.out.println(
                    "==================================================================");
            System.out.println(
                    "Start to run query "
                            + queryName
                            + " with workload "
                            + workload.getSummaryString());
            LOG.info("==================================================================");
            LOG.info(
                    "Start to run query {} with workload {}",
                    queryName,
                    workload.getSummaryString());
            runInternal();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void runInternal() throws Exception {
        Map<String, String> varsMap = initializeVarsMap();
        // These stmts will not be explained
        List<String> prepareStmts = getPreparationStmts(varsMap);
        // INSERT stmts will be explained
        List<String> queryStmts = getQueryStmts(varsMap);
        submitSQLJob(prepareStmts, queryStmts);
    }

    private Map<String, String> initializeVarsMap() {
        LocalDateTime currentTime = LocalDateTime.now();
        LocalDateTime submitTime = currentTime.minusNanos(currentTime.getNano());

        Map<String, String> varsMap = new HashMap<>();
        varsMap.put("SUBMIT_TIME", submitTime.toString());
        varsMap.put("EVENT_RATES", String.valueOf(workload.getEventRates()));
        varsMap.put("RATE_INTERVAL_SECONDS", String.valueOf(workload.getRateIntervalSeconds()));
        varsMap.put(
                "EVENTS_NUM",
                String.valueOf(workload.getKafkaServers() == null ? workload.getEventsNum() : 0));
        varsMap.put("PERSON_PROPORTION", String.valueOf(workload.getPersonProportion()));
        varsMap.put("AUCTION_PROPORTION", String.valueOf(workload.getAuctionProportion()));
        varsMap.put("BID_PROPORTION", String.valueOf(workload.getBidProportion()));
        varsMap.put("NEXMARK_TABLE", workload.getKafkaServers() == null ? "datagen" : "kafka");
        varsMap.put(
                "BOOTSTRAP_SERVERS",
                workload.getKafkaServers() == null ? "" : workload.getKafkaServers());
        varsMap.put("BASE_TIME", String.valueOf(baseTime));
        return varsMap;
    }

    private List<String> getPreparationStmts(Map<String, String> vars) throws IOException {
        List<String> allLines = new ArrayList<>();
        allLines.addAll(initializeSqlFileLines(vars, "/nexmark/ddl/ddl_gen.sql"));
        allLines.addAll(initializeSqlFileLines(vars, "/nexmark/ddl/ddl_kafka.sql"));
        allLines.addAll(initializeSqlFileLines(vars, "/nexmark/ddl/ddl_views.sql"));
        if (Objects.nonNull(workload.getKafkaServers())) {
            allLines.addAll(initializeSqlFileLines(vars, "/nexmark/ddl/insert_kafka.sql"));
        }
        return allLines;
    }

    private List<String> getQueryStmts(Map<String, String> vars) throws IOException {
        return initializeSqlFileLines(vars, "/nexmark/queries/" + queryName + ".sql");
    }

    private List<String> initializeSqlFileLines(Map<String, String> vars, String resourcePath)
            throws IOException {
        InputStream input = getClass().getResourceAsStream(resourcePath);
        if (input == null) {
            throw new IOException("Resource not found: " + resourcePath);
        }
        BufferedReader reader = new BufferedReader(new InputStreamReader(input));
        List<String> result = new ArrayList<>();

        while (true) {
            String line = reader.readLine();
            if (line == null) {
                break;
            }
            // Remove comments
            line = line.split("--", 2)[0].trim();
            for (Map.Entry<String, String> var : vars.entrySet()) {
                line = line.replace("${" + var.getKey() + "}", var.getValue());
            }

            result.add(line);
        }
        reader.close();
        input.close();

        String string = String.join(" ", result);
        String[] lines = string.split(";");
        List<String> sqlLines = new ArrayList<>();
        for (String sql : lines) {
            sql = sql.trim();
            if (!sql.isEmpty()) {
                sqlLines.add(sql + ";");
            }
        }
        return sqlLines;
    }

    private void prepareKafkaTopic(int numPartitions, String bootstrapServer) {

        if (kafkaHome == null) {
            throw new IllegalArgumentException("Kafka home is not set");
        }

        // Remove topic named 'nexmark' if exists
        String[] removeCommand = {
            kafkaHome + "/bin/kafka-topics.sh",
            "--delete",
            "--topic",
            "nexmark",
            "--if-exists",
            "--bootstrap-server",
            bootstrapServer,
        };
        executeCommand(removeCommand);

        String[] createCommand = {
            kafkaHome + "/bin/kafka-topics.sh",
            "--create",
            "--topic",
            "nexmark",
            "--partitions",
            String.valueOf(numPartitions),
            "--replication-factor",
            "1",
            "--bootstrap-server",
            bootstrapServer
        };
        executeCommand(createCommand);
    }

    private void executeCommand(String[] command) {
        System.out.println("Executing command: " + String.join(" ", command));

        try {
            AutoClosableProcess.create(command)
                    .setStdoutProcessor(LOG::info)
                    .setStderrProcessor(LOG::error)
                    .runBlocking();
        } catch (IOException e) {
            throw new RuntimeException("Failed to execute command: " + Arrays.toString(command), e);
        }
    }

    public void submitSQLJob(List<String> prepareStmts, List<String> queryStmts) throws Exception {
        StreamExecutionEnvironment sEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment env = StreamTableEnvironment.create(sEnv);

        if (!explainOnly && Objects.nonNull(workload.getKafkaServers())) {
            prepareKafkaTopic(sEnv.getParallelism(), workload.getKafkaServers());
        }

        for (String stmt : prepareStmts) {
            env.executeSql(stmt);
        }

        for (String stmt : queryStmts) {
            if (!isInsert(stmt)) {
                env.executeSql(stmt);
            } else {
                try {
                    String explanation = env.explainSql(stmt, ExplainDetail.JSON_EXECUTION_PLAN);
                    System.out.println("Execution Plan: " + explanation);
                    if (!explainOnly) {
                        env.executeSql(stmt);
                    } else {
                        HDFSUtils.write(explanation, planSaveFileName);
                    }
                } catch (Exception e) {
                    throw new RuntimeException("Failed to execute SQL statement: " + stmt, e);
                }
            }
        }
    }

    private boolean isInsert(String sql) {
        return sql.trim().toUpperCase().startsWith("INSERT");
    }
}
