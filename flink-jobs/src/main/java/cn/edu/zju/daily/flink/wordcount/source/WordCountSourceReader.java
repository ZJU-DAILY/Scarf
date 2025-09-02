package cn.edu.zju.daily.flink.wordcount.source;

import cn.edu.zju.daily.flink.source.ElasticSourceReader;
import org.apache.flink.api.connector.source.SourceReaderContext;

public class WordCountSourceReader
        extends ElasticSourceReader<TimestampedString, TimestampedString> {

    public WordCountSourceReader(WordCountGeneratorConfig config, SourceReaderContext context) {
        super(() -> new WordCountSplitReader(config), new IdentityRecordEmitter<>(), context);
    }
}
