package cn.edu.zju.daily.flink.wordcount.source;

import cn.edu.zju.daily.flink.source.RawEventGenerator;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import lombok.Getter;
import org.apache.commons.lang3.StringUtils;

public class WordCountEventGenerator extends RawEventGenerator<TimestampedString> {

    @Getter private final WordCountGeneratorConfig config;
    private final List<String> lines;

    public WordCountEventGenerator(WordCountGeneratorConfig config, Object lock) {
        super(lock);
        this.config = config;
        int maxWordsPerLine = config.getMaxWordsPerLine();
        String resourceName = config.getResourceName();
        try {
            this.lines =
                    buildLines(
                            Thread.currentThread()
                                    .getContextClassLoader()
                                    .getResourceAsStream(resourceName),
                            maxWordsPerLine);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected TimestampedString generate(long nextEventId, long nextEventTime, long splitCount) {
        String value = lines.get((int) (nextEventId % lines.size()));
        return new TimestampedString(value, nextEventTime);
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
