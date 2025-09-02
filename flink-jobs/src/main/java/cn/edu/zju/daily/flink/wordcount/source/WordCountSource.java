package cn.edu.zju.daily.flink.wordcount.source;

import cn.edu.zju.daily.flink.source.ElasticSource;
import cn.edu.zju.daily.flink.source.ElasticSourceSplit;
import org.apache.flink.api.connector.source.SourceReader;
import org.apache.flink.api.connector.source.SourceReaderContext;

public class WordCountSource extends ElasticSource<TimestampedString, TimestampedString> {

    private final WordCountGeneratorConfig config;

    public WordCountSource(WordCountGeneratorConfig config) {
        super(
                config.getMaxEvents(),
                config.getFirstEventId(),
                config.getRateIntervalSeconds() * 1000,
                config.getLocalEventDelayUs());
        this.config = config;
    }

    @Override
    public SourceReader<TimestampedString, ElasticSourceSplit> createReader(
            SourceReaderContext readerContext) {
        return new WordCountSourceReader(config, readerContext);
    }
}
