package cn.edu.zju.daily.flink.nexmark.source.v2;

import cn.edu.zju.daily.flink.nexmark.generator.NexmarkGeneratorConfig;
import cn.edu.zju.daily.flink.nexmark.model.Event;
import cn.edu.zju.daily.flink.source.ElasticSourceReader;
import org.apache.flink.api.connector.source.SourceReaderContext;
import org.apache.flink.table.data.RowData;

public class NexmarkSourceReader extends ElasticSourceReader<Event, RowData> {

    public NexmarkSourceReader(
            NexmarkGeneratorConfig generatorConfig, SourceReaderContext context) {
        super(() -> new NexmarkSplitReader(generatorConfig), new NexmarkRecordEmitter(), context);
    }
}
