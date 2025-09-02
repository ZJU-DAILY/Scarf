package cn.edu.zju.daily.flink.nexmark.source.v2;

import static org.apache.flink.table.api.DataTypes.*;

import cn.edu.zju.daily.flink.nexmark.generator.NexmarkGeneratorConfig;
import java.util.Objects;
import java.util.stream.Collectors;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.connector.ChangelogMode;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.connector.source.ScanTableSource;
import org.apache.flink.table.connector.source.SourceProvider;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.types.DataType;

/** Table source for Nexmark. */
public class NexmarkTableSource implements ScanTableSource {

    public static final Schema NEXMARK_SCHEMA =
            Schema.newBuilder()
                    .column("event_type", INT())
                    .column(
                            "person",
                            ROW(
                                    FIELD("id", BIGINT()),
                                    FIELD("name", STRING()),
                                    FIELD("emailAddress", STRING()),
                                    FIELD("creditCard", STRING()),
                                    FIELD("city", STRING()),
                                    FIELD("state", STRING()),
                                    FIELD("dateTime", TIMESTAMP(3)),
                                    FIELD("extra", STRING())))
                    .column(
                            "auction",
                            ROW(
                                    FIELD("id", BIGINT()),
                                    FIELD("itemName", STRING()),
                                    FIELD("description", STRING()),
                                    FIELD("initialBid", BIGINT()),
                                    FIELD("reserve", BIGINT()),
                                    FIELD("dateTime", TIMESTAMP(3)),
                                    FIELD("expires", TIMESTAMP(3)),
                                    FIELD("seller", BIGINT()),
                                    FIELD("category", BIGINT()),
                                    FIELD("extra", STRING())))
                    .column(
                            "bid",
                            ROW(
                                    FIELD("auction", BIGINT()),
                                    FIELD("bidder", BIGINT()),
                                    FIELD("price", BIGINT()),
                                    FIELD("channel", STRING()),
                                    FIELD("url", STRING()),
                                    FIELD("dateTime", TIMESTAMP(3)),
                                    FIELD("extra", STRING())))
                    .build();

    public static final ResolvedSchema RESOLVED_SCHEMA =
            ResolvedSchema.physical(
                    NEXMARK_SCHEMA.getColumns().stream()
                            .map(Schema.UnresolvedColumn::getName)
                            .collect(Collectors.toList()),
                    NEXMARK_SCHEMA.getColumns().stream()
                            .map(
                                    unresolvedColumn ->
                                            (DataType)
                                                    ((Schema.UnresolvedPhysicalColumn)
                                                                    unresolvedColumn)
                                                            .getDataType())
                            .collect(Collectors.toList()));

    private final NexmarkGeneratorConfig config;

    public NexmarkTableSource(NexmarkGeneratorConfig config) {
        this.config = config;
    }

    @Override
    public ChangelogMode getChangelogMode() {
        return ChangelogMode.insertOnly();
    }

    @Override
    public ScanRuntimeProvider getScanRuntimeProvider(ScanContext scanContext) {
        TypeInformation<RowData> outputType =
                scanContext.createTypeInformation(RESOLVED_SCHEMA.toPhysicalRowDataType());

        // Step 1: Create the DataStream from the source
        return SourceProvider.of(new NexmarkSource(config));
    }

    @Override
    public DynamicTableSource copy() {
        return new NexmarkTableSource(config);
    }

    @Override
    public String asSummaryString() {
        return "Nexmark Source";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        NexmarkTableSource that = (NexmarkTableSource) o;
        return Objects.equals(config, that.config);
    }

    @Override
    public int hashCode() {
        return Objects.hash(config);
    }

    @Override
    public String toString() {
        return "NexmarkTableSource{" + "config=" + config + '}';
    }
}
