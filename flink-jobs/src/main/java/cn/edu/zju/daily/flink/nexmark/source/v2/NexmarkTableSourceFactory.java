package cn.edu.zju.daily.flink.nexmark.source.v2;

import cn.edu.zju.daily.flink.nexmark.NexmarkConfiguration;
import cn.edu.zju.daily.flink.nexmark.generator.NexmarkGeneratorConfig;
import cn.edu.zju.daily.flink.nexmark.source.NexmarkSourceOptions;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import lombok.extern.slf4j.Slf4j;
import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.CoreOptions;
import org.apache.flink.configuration.ReadableConfig;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.factories.DynamicTableSourceFactory;
import org.apache.flink.table.factories.FactoryUtil;

@Slf4j
public class NexmarkTableSourceFactory implements DynamicTableSourceFactory {

    @Override
    public DynamicTableSource createDynamicTableSource(Context context) {
        final FactoryUtil.TableFactoryHelper helper =
                FactoryUtil.createTableFactoryHelper(this, context);
        final ReadableConfig config = helper.getOptions();
        helper.validate();

        // for compatibility reason of "context.getCatalogTable()", do not validate schema.
        // validateSchema(TableSchemaUtils.getPhysicalSchema(context.getCatalogTable().getSchema()));

        int parallelism = context.getConfiguration().get(CoreOptions.DEFAULT_PARALLELISM);
        NexmarkConfiguration nexmarkConf =
                NexmarkSourceOptions.convertToNexmarkConfiguration(config);
        nexmarkConf.numEventGenerators = NexmarkSource.NUM_SPLITS;
        NexmarkGeneratorConfig generatorConfig =
                new NexmarkGeneratorConfig(
                        nexmarkConf, nexmarkConf.baseTime, 1, nexmarkConf.numEvents, 1);

        LOG.info("Nexmark source config: {}", generatorConfig);

        return new NexmarkTableSource(generatorConfig);
    }

    @Override
    public String factoryIdentifier() {
        return "nexmark-v2";
    }

    @Override
    public Set<ConfigOption<?>> requiredOptions() {
        return Collections.emptySet();
    }

    @Override
    public Set<ConfigOption<?>> optionalOptions() {
        Set<ConfigOption<?>> sets = new HashSet<>();
        sets.add(NexmarkSourceOptions.RATE_LIMITED);
        sets.add(NexmarkSourceOptions.EVENT_RATES);
        sets.add(NexmarkSourceOptions.RATE_INTERVAL_SECONDS);
        sets.add(NexmarkSourceOptions.PERSON_AVG_SIZE);
        sets.add(NexmarkSourceOptions.AUCTION_AVG_SIZE);
        sets.add(NexmarkSourceOptions.BID_AVG_SIZE);
        sets.add(NexmarkSourceOptions.PERSON_PROPORTION);
        sets.add(NexmarkSourceOptions.AUCTION_PROPORTION);
        sets.add(NexmarkSourceOptions.BID_PROPORTION);
        sets.add(NexmarkSourceOptions.BID_HOT_RATIO_AUCTIONS);
        sets.add(NexmarkSourceOptions.BID_HOT_RATIO_BIDDERS);
        sets.add(NexmarkSourceOptions.AUCTION_HOT_RATIO_SELLERS);
        sets.add(NexmarkSourceOptions.EVENTS_NUM);
        sets.add(NexmarkSourceOptions.BASE_TIME);
        return sets;
    }
}
