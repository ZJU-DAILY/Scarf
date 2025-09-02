package cn.edu.zju.daily.flink.yahoo.source;

import cn.edu.zju.daily.flink.source.ElasticSplitReader;
import org.apache.flink.api.java.tuple.Tuple7;

public class YahooSplitReader
        extends ElasticSplitReader<Tuple7<String, String, String, String, String, Long, String>> {

    public YahooSplitReader(YahooConfig config) {
        this(config, new Object());
    }

    public YahooSplitReader(YahooConfig config, Object lock) {
        super(new YahooEventGenerator(config, lock), lock, config.getBaseTime());
    }
}
