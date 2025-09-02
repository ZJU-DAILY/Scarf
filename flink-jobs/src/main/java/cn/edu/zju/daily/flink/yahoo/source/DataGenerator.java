package cn.edu.zju.daily.flink.yahoo.source;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import org.apache.flink.api.java.tuple.Tuple7;

/** DataGenerator - generates campaign/ad IDs and random ad event JSONs. */
public class DataGenerator {

    // Configuration constants
    public static final int NUM_USERS = 10000;
    public static final int NUM_PAGES = 1000;
    public static final int NUM_CAMPAIGNS = 5000;
    public static final int ADS_PER_CAMPAIGN = 10;
    public static final int NUM_ADS = NUM_CAMPAIGNS * ADS_PER_CAMPAIGN;

    private static final Random RANDOM = new Random();

    /** Generates a list of random UUID strings. */
    public static List<String> makeIds(int n) {
        return IntStream.range(0, n)
                .mapToObj(i -> UUID.randomUUID().toString())
                .collect(Collectors.toList());
    }

    /**
     * Generates a random ad event as a JSON string. Optionally adds skew and lateness to the
     * event_time.
     */
    public static Tuple7<String, String, String, String, String, Long, String> makeAdEventAt(
            long time,
            boolean withSkew,
            List<String> ads,
            List<String> userIds,
            List<String> pageIds) {
        String[] adTypes = {"banner", "modal", "sponsored-search", "mail", "mobile"};
        String[] eventTypes = {"view", "click", "purchase"};

        int skew = withSkew ? (50 - RANDOM.nextInt(100)) : 0;
        int lateBy = 0;
        if (withSkew) {
            if (RANDOM.nextInt(100_000) == 0) {
                lateBy = RANDOM.nextInt(60_000);
            }
            lateBy = -lateBy;
        }
        long eventTime = time + skew + lateBy;

        try {
            return Tuple7.of(
                    randomFrom(userIds),
                    randomFrom(pageIds),
                    randomFrom(ads),
                    randomFrom(adTypes),
                    randomFrom(eventTypes),
                    eventTime,
                    "1.2.3.4");
        } catch (Exception e) {
            throw new RuntimeException("Error generating JSON", e);
        }
    }

    /** Utility to pick a random element from a list or array. */
    public static <T> T randomFrom(List<T> list) {
        return list.get(RANDOM.nextInt(list.size()));
    }

    public static <T> T randomFrom(T[] array) {
        return array[RANDOM.nextInt(array.length)];
    }
}
