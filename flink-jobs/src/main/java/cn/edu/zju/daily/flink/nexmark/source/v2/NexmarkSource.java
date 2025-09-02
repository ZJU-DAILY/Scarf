package cn.edu.zju.daily.flink.nexmark.source.v2;

import cn.edu.zju.daily.flink.nexmark.generator.NexmarkGeneratorConfig;
import cn.edu.zju.daily.flink.nexmark.model.Event;
import cn.edu.zju.daily.flink.source.ElasticSource;
import cn.edu.zju.daily.flink.source.ElasticSourceSplit;
import lombok.extern.slf4j.Slf4j;
import org.apache.flink.api.connector.source.*;
import org.apache.flink.table.data.RowData;

@Slf4j
public class NexmarkSource extends ElasticSource<Event, RowData> {

    private final NexmarkGeneratorConfig config;

    public NexmarkSource(NexmarkGeneratorConfig config) {
        super(
                config.maxEvents,
                config.firstEventId,
                config.getStepLengthSec() * 1000,
                config.getLocalEventDelayUs());
        this.config = config;
    }

    @Override
    public SourceReader<RowData, ElasticSourceSplit> createReader(SourceReaderContext readerContext)
            throws Exception {
        return new NexmarkSourceReader(config, readerContext);
    }
}
