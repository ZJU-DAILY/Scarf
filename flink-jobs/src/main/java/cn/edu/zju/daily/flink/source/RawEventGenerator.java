package cn.edu.zju.daily.flink.source;

public abstract class RawEventGenerator<E> {

    private final Object lock;

    protected RawEventGenerator(Object lock) {
        this.lock = lock;
    }

    protected abstract E generate(long nextEventId, long nextEventTime, long splitCount);

    public E next(ElasticSourceSplit split) throws InterruptedException {
        // Get the split that has the earliest next event time
        long nextEventTime = split.getNextEventTimeMillis();
        long nextEmitTime = split.getNextEmitTimeMillis();
        long nextEventId = split.getNextEventId();

        E event = generate(nextEventId, nextEventTime, split.getOffsetEventId());

        long now = System.currentTimeMillis();
        if (nextEmitTime > now) {
            long delay = nextEmitTime - now;
            synchronized (lock) {
                lock.wait(delay);
            }
        }
        split.incrementCount();

        return event;
    }
}
