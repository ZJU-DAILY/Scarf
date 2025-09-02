package cn.edu.zju.daily.flink.yahoo.source;

import cn.edu.zju.daily.flink.source.RawEventGenerator;
import org.apache.flink.api.java.tuple.Tuple7;

public class YahooEventGenerator
        extends RawEventGenerator<Tuple7<String, String, String, String, String, Long, String>> {

    private final YahooConfig config;

    public YahooEventGenerator(YahooConfig config, Object lock) {
        super(lock);
        this.config = config;
    }

    @Override
    protected Tuple7<String, String, String, String, String, Long, String> generate(
            long nextEventId, long nextEventTime, long splitCount) {
        return DataGenerator.makeAdEventAt(
                nextEventTime, false, config.getAdIds(), config.getUserIds(), config.getPageIds());
    }
}
