package cn.edu.zju.daily.flink.nexmark.source.v2;

import cn.edu.zju.daily.flink.nexmark.generator.NexmarkGeneratorConfig;
import cn.edu.zju.daily.flink.nexmark.model.Event;
import cn.edu.zju.daily.flink.source.ElasticSplitReader;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class NexmarkSplitReader extends ElasticSplitReader<Event> {

    private NexmarkSplitReader(NexmarkGeneratorConfig config, Object lock) {
        super(new NexmarkEventGenerator(config, lock), lock, config.getBaseTime());
    }

    public NexmarkSplitReader(NexmarkGeneratorConfig config) {
        this(config, new Object());
    }
}
