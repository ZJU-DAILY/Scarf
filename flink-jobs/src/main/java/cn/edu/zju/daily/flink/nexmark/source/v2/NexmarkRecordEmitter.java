package cn.edu.zju.daily.flink.nexmark.source.v2;

import cn.edu.zju.daily.flink.nexmark.model.Event;
import cn.edu.zju.daily.flink.nexmark.source.RowDataEventDeserializer;
import cn.edu.zju.daily.flink.source.ElasticSourceSplitState;
import org.apache.flink.api.connector.source.SourceOutput;
import org.apache.flink.connector.base.source.reader.RecordEmitter;
import org.apache.flink.table.data.RowData;

public class NexmarkRecordEmitter
        implements RecordEmitter<Event, RowData, ElasticSourceSplitState> {

    RowDataEventDeserializer deserializer = new RowDataEventDeserializer();

    @Override
    public void emitRecord(
            Event element, SourceOutput<RowData> output, ElasticSourceSplitState splitState)
            throws Exception {
        output.collect(deserializer.deserialize(element));
    }
}
