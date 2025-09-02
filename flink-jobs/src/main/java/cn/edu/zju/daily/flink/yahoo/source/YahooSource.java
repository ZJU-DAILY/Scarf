package cn.edu.zju.daily.flink.yahoo.source;

import cn.edu.zju.daily.flink.source.ElasticSource;
import cn.edu.zju.daily.flink.source.ElasticSourceSplit;
import org.apache.flink.api.connector.source.SourceReader;
import org.apache.flink.api.connector.source.SourceReaderContext;
import org.apache.flink.api.java.tuple.Tuple7;

public class YahooSource
        extends ElasticSource<
                Tuple7<String, String, String, String, String, Long, String>,
                Tuple7<String, String, String, String, String, Long, String>> {

    private final YahooConfig config;

    public YahooSource(YahooConfig config) {
        super(
                config.getMaxEvents(),
                config.getFirstEventId(),
                config.getRateIntervalSeconds() * 1000,
                config.getLocalEventDelayUs());
        this.config = config;
    }

    @Override
    public SourceReader<
                    Tuple7<String, String, String, String, String, Long, String>,
                    ElasticSourceSplit>
            createReader(SourceReaderContext readerContext) throws Exception {
        return new YahooSourceReader(config, readerContext);
    }
}
