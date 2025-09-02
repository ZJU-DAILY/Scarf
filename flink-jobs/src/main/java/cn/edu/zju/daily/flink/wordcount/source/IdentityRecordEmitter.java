package cn.edu.zju.daily.flink.wordcount.source;

import cn.edu.zju.daily.flink.source.ElasticSourceSplitState;
import org.apache.flink.api.connector.source.SourceOutput;
import org.apache.flink.connector.base.source.reader.RecordEmitter;

public class IdentityRecordEmitter<T> implements RecordEmitter<T, T, ElasticSourceSplitState> {

    @Override
    public void emitRecord(T element, SourceOutput<T> output, ElasticSourceSplitState splitState) {
        output.collect(element);
    }
}
