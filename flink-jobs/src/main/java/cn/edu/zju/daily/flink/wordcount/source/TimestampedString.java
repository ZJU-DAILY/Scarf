package cn.edu.zju.daily.flink.wordcount.source;

import java.io.Serializable;
import lombok.Data;

@Data
public class TimestampedString implements Serializable {

    private final String value;
    private final long timestamp;
}
