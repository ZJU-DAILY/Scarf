package cn.edu.zju.daily.flink.nexmark.source.v2;

import cn.edu.zju.daily.flink.nexmark.generator.NexmarkGeneratorConfig;
import cn.edu.zju.daily.flink.nexmark.generator.model.AuctionGenerator;
import cn.edu.zju.daily.flink.nexmark.generator.model.BidGenerator;
import cn.edu.zju.daily.flink.nexmark.generator.model.PersonGenerator;
import cn.edu.zju.daily.flink.nexmark.model.Event;
import cn.edu.zju.daily.flink.source.RawEventGenerator;
import java.util.SplittableRandom;
import lombok.Getter;

public class NexmarkEventGenerator extends RawEventGenerator<Event> {

    @Getter private final NexmarkGeneratorConfig config;
    private final SplittableRandom random = new SplittableRandom();

    public NexmarkEventGenerator(NexmarkGeneratorConfig config, Object lock) {
        super(lock);
        this.config = config;
    }

    @Override
    protected Event generate(long nextEventId, long nextEventTime, long splitCount) {
        long rem = nextEventId % config.totalProportion;
        if (rem < config.personProportion) {
            return new Event(
                    PersonGenerator.nextPerson(nextEventId, random, nextEventTime, config));
        } else if (rem < config.personProportion + config.auctionProportion) {
            return new Event(
                    AuctionGenerator.nextAuction(
                            splitCount, nextEventId, random, nextEventTime, config));
        } else {
            return new Event(BidGenerator.nextBid(nextEventId, random, nextEventTime, config));
        }
    }
}
