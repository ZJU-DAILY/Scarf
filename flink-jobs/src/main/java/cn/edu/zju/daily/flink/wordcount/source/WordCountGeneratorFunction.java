package cn.edu.zju.daily.flink.wordcount.source;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.lang3.StringUtils;
import org.apache.flink.api.connector.source.SourceReaderContext;
import org.apache.flink.connector.datagen.source.GeneratorFunction;
import org.apache.flink.metrics.Counter;

@Deprecated
public class WordCountGeneratorFunction implements GeneratorFunction<Long, String>, Serializable {

    private final int maxWordsPerLine;
    private final String resourceName;
    private List<String> lines;
    private Counter numRecordsInCounter;

    public WordCountGeneratorFunction(int maxWordsPerLine, String resourceName) {
        this.maxWordsPerLine = maxWordsPerLine;
        this.resourceName = resourceName;
    }

    @Override
    public void open(SourceReaderContext readerContext) throws Exception {
        lines =
                buildLines(
                        Thread.currentThread()
                                .getContextClassLoader()
                                .getResourceAsStream(resourceName),
                        maxWordsPerLine);
        numRecordsInCounter =
                readerContext.metricGroup().getIOMetricGroup().getNumRecordsInCounter();
    }

    @Override
    public void close() throws Exception {}

    @Override
    public String map(Long value) throws Exception {
        if (lines.isEmpty()) {
            throw new IllegalStateException("No lines available.");
        }
        numRecordsInCounter.inc();
        return lines.get((int) (value % lines.size()));
    }

    private static List<String> buildLines(InputStream is, int maxWordsPerLine) throws IOException {
        List<String> lines = new ArrayList<>();
        StringBuilder builder = new StringBuilder();
        try (is) {
            String line;
            try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
                while ((line = br.readLine()) != null) {
                    line = StringUtils.stripAccents(line.strip());
                    if (line.isEmpty()) {
                        add(lines, builder.toString().strip(), maxWordsPerLine);
                        builder.delete(0, builder.length());
                    } else {
                        String stripped = line.replaceAll("[^\\w\\s]+", " ").toLowerCase().strip();
                        builder.append(stripped).append(" ");
                    }
                }
            }
        }
        String remaining = builder.toString().strip();
        if (!remaining.isEmpty()) {
            add(lines, remaining, maxWordsPerLine);
        }
        if (lines.isEmpty()) {
            throw new IllegalArgumentException("No lines available.");
        }
        System.out.println("Number of unique lines: " + lines.size());
        return lines;
    }

    private static void add(List<String> list, String line, int maxWordsPerLine) {
        line = line.strip();
        if (line.isEmpty()) {
            list.add("");
            return;
        }
        String[] words = line.split("\\s+");
        for (int i = 0; i < words.length; i += maxWordsPerLine) {
            StringBuilder builder = new StringBuilder();
            for (int j = i; j < i + maxWordsPerLine && j < words.length; j++) {
                builder.append(words[j]).append(" ");
            }
            list.add(builder.toString().strip());
        }
    }
}
