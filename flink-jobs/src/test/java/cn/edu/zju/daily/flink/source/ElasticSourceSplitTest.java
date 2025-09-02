package cn.edu.zju.daily.flink.source;

import org.junit.Test;

public class ElasticSourceSplitTest {

    @Test
    public void test() {
        ElasticSourceSplit split =
                new ElasticSourceSplit(4, 10, 3120, 10000, 1000, new double[] {1000.0D, 2000.0D});

        split.setBaseEmitTime(1000000000);
        split.setOffsetEventId(1_000);

        long nextEventId = split.getNextEventId();
        long nextEventTimeMillis = split.getNextEventTimeMillis();

        System.out.println(nextEventId);
        System.out.println(nextEventTimeMillis);

        assert (nextEventId == 3120 + 1000);
        assert (nextEventTimeMillis == 1000000000 + 6 * 2000 + 1000);
    }
}
