CREATE TABLE nexmark_q3a (
  bidder_name VARCHAR,
  bid_price BIGINT,
  auction_id BIGINT
) WITH (
  'connector' = 'blackhole'
);

INSERT INTO nexmark_q3a
SELECT
    P.name AS bidder_name,
    B.price AS bid_price,
    B.auction AS auction_id
FROM
    bid AS B INNER JOIN person AS P ON B.bidder = P.id
WHERE
    B.price > 1000
    AND (P.state = 'WA' OR P.state = 'OR');
