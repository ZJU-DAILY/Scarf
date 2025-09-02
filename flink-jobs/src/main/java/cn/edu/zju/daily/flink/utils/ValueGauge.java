package cn.edu.zju.daily.flink.utils;

import org.apache.flink.metrics.Gauge;

public class ValueGauge<T> implements Gauge<T> {

    private T value;

    public ValueGauge(T initialValue) {
        this.value = initialValue;
    }

    public void setValue(T value) {
        this.value = value;
    }

    @Override
    public T getValue() {
        return null;
    }
}
