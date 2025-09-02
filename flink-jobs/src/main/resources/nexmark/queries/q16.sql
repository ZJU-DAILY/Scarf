-- -------------------------------------------------------------------------------------------------
-- Query 16: Channel Statistics Report (Not in original suite)
-- -------------------------------------------------------------------------------------------------
-- How many distinct users join the bidding for different level of price for a channel?
-- Illustrates multiple distinct aggregations with filters for multiple keys.
-- -------------------------------------------------------------------------------------------------

CREATE TABLE nexmark_q16
(
    channel        VARCHAR,
    `day`          VARCHAR,
    `minute`       VARCHAR,
    total_bids     BIGINT,
    rank1_bids     BIGINT,
    rank2_bids     BIGINT,
    rank3_bids     BIGINT,
    total_bidders  BIGINT,
    rank1_bidders  BIGINT,
    rank2_bidders  BIGINT,
    rank3_bidders  BIGINT,
    total_auctions BIGINT,
    rank1_auctions BIGINT,
    rank2_auctions BIGINT,
    rank3_auctions BIGINT
) WITH (
      'connector' = 'blackhole'
      );

-- INSERT INTO nexmark_q16
-- SELECT channel,
--        DATE_FORMAT(`dateTime`, 'yyyy-MM-dd') as `day`,
--        max(DATE_FORMAT(`dateTime`, 'HH:mm')) as `minute`,
--        count(*)                              AS total_bids,
--        count(*)                                 filter (where price < 10000) AS rank1_bids, count(*) filter (where price >= 10000 and price < 1000000) AS rank2_bids, count(*) filter (where price >= 1000000) AS rank3_bids, count(distinct bidder) AS total_bidders,
--        count(distinct bidder)                   filter (where price < 10000) AS rank1_bidders, count(distinct bidder) filter (where price >= 10000 and price < 1000000) AS rank2_bidders, count(distinct bidder) filter (where price >= 1000000) AS rank3_bidders, count(distinct auction) AS total_auctions,
--        count(distinct auction)                  filter (where price < 10000) AS rank1_auctions, count(distinct auction) filter (where price >= 10000 and price < 1000000) AS rank2_auctions, count(distinct auction) filter (where price >= 1000000) AS rank3_auctions
-- FROM bid
-- GROUP BY channel, DATE_FORMAT(`dateTime`, 'yyyy-MM-dd');

-- We add a 1-minute tumbling window to make it more suitable for continuous processing.
INSERT INTO nexmark_q16
SELECT
    channel,
    -- window start as YYYY-MM-dd
    DATE_FORMAT(TUMBLE_START(`dateTime`, INTERVAL '1' MINUTE), 'yyyy-MM-dd')  AS `day`,
    -- window start as HH:mm
    DATE_FORMAT(TUMBLE_START(`dateTime`, INTERVAL '1' MINUTE), 'HH:mm')       AS `minute`,
    COUNT(*)                                                                  AS total_bids,
    COUNT(*) FILTER (WHERE price < 10000)                                     AS rank1_bids,
    COUNT(*) FILTER (WHERE price >= 10000 AND price < 1000000)                AS rank2_bids,
    COUNT(*) FILTER (WHERE price >= 1000000)                                  AS rank3_bids,
    COUNT(DISTINCT bidder)                                                    AS total_bidders,
    COUNT(DISTINCT bidder) FILTER (WHERE price < 10000)                       AS rank1_bidders,
    COUNT(DISTINCT bidder) FILTER (WHERE price >= 10000 AND price < 1000000)  AS rank2_bidders,
    COUNT(DISTINCT bidder) FILTER (WHERE price >= 1000000)                    AS rank3_bidders,
    COUNT(DISTINCT auction)                                                   AS total_auctions,
    COUNT(DISTINCT auction) FILTER (WHERE price < 10000)                      AS rank1_auctions,
    COUNT(DISTINCT auction) FILTER (WHERE price >= 10000 AND price < 1000000) AS rank2_auctions,
    COUNT(DISTINCT auction) FILTER (WHERE price >= 1000000)                   AS rank3_auctions
FROM bid
GROUP BY channel, TUMBLE(`dateTime`, INTERVAL '1' MINUTE);