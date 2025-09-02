package cn.edu.zju.daily.flink.wordcount.source;

import cn.edu.zju.daily.flink.source.ElasticSplitReader;

public class WordCountSplitReader extends ElasticSplitReader<TimestampedString> {

    public WordCountSplitReader(WordCountGeneratorConfig config) {
        this(config, new Object());
    }

    public WordCountSplitReader(WordCountGeneratorConfig config, Object lock) {
        super(new WordCountEventGenerator(config, lock), lock, config.getBaseTime());
    }
}
