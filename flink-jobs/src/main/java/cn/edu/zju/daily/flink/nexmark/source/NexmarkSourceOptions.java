/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cn.edu.zju.daily.flink.nexmark.source;

import cn.edu.zju.daily.flink.nexmark.NexmarkConfiguration;
import cn.edu.zju.daily.flink.nexmark.utils.NexmarkUtils;
import java.time.Duration;
import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.ConfigOptions;
import org.apache.flink.configuration.MemorySize;
import org.apache.flink.configuration.ReadableConfig;

public class NexmarkSourceOptions {

    /**
     * @see NexmarkConfiguration#rateShape
     */
    public static final ConfigOption<NexmarkUtils.RateShape> RATE_SHAPE =
            ConfigOptions.key("rate.shape")
                    .enumType(NexmarkUtils.RateShape.class)
                    .defaultValue(NexmarkUtils.RateShape.SQUARE);

    /**
     * @see NexmarkConfiguration#rateIntervalSec
     */
    public static final ConfigOption<Duration> RATE_PERIOD =
            ConfigOptions.key("rate.period").durationType().defaultValue(Duration.ofSeconds(600));

    /**
     * @see NexmarkConfiguration#isRateLimited
     */
    public static final ConfigOption<Boolean> RATE_LIMITED =
            ConfigOptions.key("rate.limited").booleanType().defaultValue(false);

    /**
     * @see NexmarkConfiguration#eventRates
     */
    public static final ConfigOption<String> EVENT_RATES =
            ConfigOptions.key("event-rates").stringType().noDefaultValue();

    /**
     * @see NexmarkConfiguration#rateIntervalSec
     */
    public static final ConfigOption<Integer> RATE_INTERVAL_SECONDS =
            ConfigOptions.key("rate-interval-seconds")
                    .intType()
                    .defaultValue(1)
                    .withDescription("The interval in seconds for the change rate of events. ");

    /**
     * @see NexmarkConfiguration#avgPersonByteSize
     */
    public static final ConfigOption<MemorySize> PERSON_AVG_SIZE =
            ConfigOptions.key("person.avg-size")
                    .memoryType()
                    .defaultValue(MemorySize.parse("200b"));

    /**
     * @see NexmarkConfiguration#avgAuctionByteSize
     */
    public static final ConfigOption<MemorySize> AUCTION_AVG_SIZE =
            ConfigOptions.key("auction.avg-size")
                    .memoryType()
                    .defaultValue(MemorySize.parse("500b"));

    /**
     * @see NexmarkConfiguration#avgBidByteSize
     */
    public static final ConfigOption<MemorySize> BID_AVG_SIZE =
            ConfigOptions.key("bid.avg-size").memoryType().defaultValue(MemorySize.parse("100b"));

    /**
     * @see NexmarkConfiguration#personProportion
     */
    public static final ConfigOption<Integer> PERSON_PROPORTION =
            ConfigOptions.key("person.proportion").intType().defaultValue(1);

    /**
     * @see NexmarkConfiguration#auctionProportion
     */
    public static final ConfigOption<Integer> AUCTION_PROPORTION =
            ConfigOptions.key("auction.proportion").intType().defaultValue(3);

    /**
     * @see NexmarkConfiguration#bidProportion
     */
    public static final ConfigOption<Integer> BID_PROPORTION =
            ConfigOptions.key("bid.proportion").intType().defaultValue(46);

    /**
     * @see NexmarkConfiguration#hotAuctionRatio
     */
    public static final ConfigOption<Integer> BID_HOT_RATIO_AUCTIONS =
            ConfigOptions.key("bid.hot-ratio.auctions").intType().defaultValue(2);

    /**
     * @see NexmarkConfiguration#hotBiddersRatio
     */
    public static final ConfigOption<Integer> BID_HOT_RATIO_BIDDERS =
            ConfigOptions.key("bid.hot-ratio.bidders").intType().defaultValue(4);

    /**
     * @see NexmarkConfiguration#hotSellersRatio
     */
    public static final ConfigOption<Integer> AUCTION_HOT_RATIO_SELLERS =
            ConfigOptions.key("auction.hot-ratio.sellers").intType().defaultValue(4);

    /**
     * @see NexmarkConfiguration#numEvents
     */
    public static final ConfigOption<Long> EVENTS_NUM =
            ConfigOptions.key("events.num").longType().defaultValue(0L);

    public static final ConfigOption<Long> BASE_TIME =
            ConfigOptions.key("base-time")
                    .longType()
                    .defaultValue(0L)
                    .withDescription(
                            "The base time for the Nexmark source. "
                                    + "It is used to calculate the event time of the generated events. "
                                    + "If set to 0, the event time will be the current system time.");

    public static NexmarkConfiguration convertToNexmarkConfiguration(ReadableConfig config) {
        NexmarkConfiguration nexmarkConf = new NexmarkConfiguration();
        nexmarkConf.rateShape = config.get(RATE_SHAPE);
        nexmarkConf.rateIntervalSec = config.get(RATE_INTERVAL_SECONDS);
        nexmarkConf.isRateLimited = config.get(RATE_LIMITED);
        nexmarkConf.eventRates = config.get(EVENT_RATES);
        nexmarkConf.avgPersonByteSize = (int) config.get(PERSON_AVG_SIZE).getBytes();
        nexmarkConf.avgAuctionByteSize = (int) config.get(AUCTION_AVG_SIZE).getBytes();
        nexmarkConf.avgBidByteSize = (int) config.get(BID_AVG_SIZE).getBytes();
        nexmarkConf.personProportion = config.get(PERSON_PROPORTION);
        nexmarkConf.auctionProportion = config.get(AUCTION_PROPORTION);
        nexmarkConf.bidProportion = config.get(BID_PROPORTION);
        nexmarkConf.hotAuctionRatio = config.get(BID_HOT_RATIO_AUCTIONS);
        nexmarkConf.hotBiddersRatio = config.get(BID_HOT_RATIO_BIDDERS);
        nexmarkConf.hotSellersRatio = config.get(AUCTION_HOT_RATIO_SELLERS);
        nexmarkConf.numEvents = config.get(EVENTS_NUM);
        nexmarkConf.baseTime = config.get(BASE_TIME);

        return nexmarkConf;
    }
}
