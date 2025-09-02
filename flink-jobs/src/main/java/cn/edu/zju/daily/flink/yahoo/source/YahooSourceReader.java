package cn.edu.zju.daily.flink.yahoo.source;

import cn.edu.zju.daily.flink.source.ElasticSourceReader;
import cn.edu.zju.daily.flink.wordcount.source.IdentityRecordEmitter;
import org.apache.flink.api.connector.source.SourceReaderContext;
import org.apache.flink.api.java.tuple.Tuple7;

public class YahooSourceReader
        extends ElasticSourceReader<
                Tuple7<String, String, String, String, String, Long, String>,
                Tuple7<String, String, String, String, String, Long, String>> {

    public YahooSourceReader(YahooConfig config, SourceReaderContext context) {
        super(() -> new YahooSplitReader(config), new IdentityRecordEmitter<>(), context);
    }
}
