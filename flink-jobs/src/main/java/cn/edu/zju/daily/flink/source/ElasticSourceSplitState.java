package cn.edu.zju.daily.flink.source;

import lombok.Data;

@Data
public final class ElasticSourceSplitState {

    private final ElasticSourceSplit split;
}
