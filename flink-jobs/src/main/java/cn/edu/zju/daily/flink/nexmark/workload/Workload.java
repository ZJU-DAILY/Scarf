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

package cn.edu.zju.daily.flink.nexmark.workload;

import cn.edu.zju.daily.flink.nexmark.metric.BenchmarkMetric;
import java.util.Objects;
import javax.annotation.Nullable;
import lombok.Getter;

@Getter
public class Workload {

    private final String eventRates;
    private final int rateIntervalSeconds;
    private final long eventsNum;
    private final int personProportion;
    private final int auctionProportion;
    private final int bidProportion;
    private final @Nullable String kafkaServers;

    public Workload(
            String eventRates,
            int rateIntervalSeconds,
            long eventsNum,
            int personProportion,
            int auctionProportion,
            int bidProportion) {
        this(
                eventRates,
                rateIntervalSeconds,
                eventsNum,
                personProportion,
                auctionProportion,
                bidProportion,
                null);
    }

    public Workload(
            String eventRates,
            int rateIntervalSeconds,
            long eventsNum,
            int personProportion,
            int auctionProportion,
            int bidProportion,
            @Nullable String kafkaServers) {
        this.eventRates = eventRates;
        this.rateIntervalSeconds = rateIntervalSeconds;
        this.eventsNum = eventsNum;
        this.personProportion = personProportion;
        this.auctionProportion = auctionProportion;
        this.bidProportion = bidProportion;
        this.kafkaServers = kafkaServers;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Workload workload = (Workload) o;
        return Objects.equals(eventRates, workload.eventRates)
                && rateIntervalSeconds == workload.rateIntervalSeconds
                && eventsNum == workload.eventsNum
                && personProportion == workload.personProportion
                && auctionProportion == workload.auctionProportion
                && bidProportion == workload.bidProportion
                && Objects.equals(kafkaServers, workload.kafkaServers);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                eventRates,
                rateIntervalSeconds,
                eventsNum,
                personProportion,
                auctionProportion,
                bidProportion,
                kafkaServers);
    }

    public String getSummaryString() {
        return String.format(
                "[eventRates=%s, rateIntervalSeconds=%d, eventsNum=%s, percentage=bid:%s,auction:%s,person:%s,kafkaServers:%s]",
                eventRates,
                rateIntervalSeconds,
                BenchmarkMetric.formatLongValue(eventsNum),
                bidProportion,
                auctionProportion,
                personProportion,
                kafkaServers);
    }

    @Override
    public String toString() {
        return "Workload{"
                + "eventRates='"
                + eventRates
                + "', rateIntervalSeconds="
                + rateIntervalSeconds
                + ", eventsNum="
                + eventsNum
                + ", personProportion="
                + personProportion
                + ", auctionProportion="
                + auctionProportion
                + ", bidProportion="
                + bidProportion
                + ", kafkaServers="
                + kafkaServers
                + '}';
    }
}
