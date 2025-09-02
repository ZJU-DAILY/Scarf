import json
from unittest import TestCase

from flink.sql_plan import generate_plan_hash


class TestSqlPlan(TestCase):

    plan = """{
  "nodes" : [ {
    "id" : 12,
    "type" : "Source: kafka[7]",
    "pact" : "Data Source",
    "contents" : "[7]:TableSourceScan(table=[[default_catalog, default_database, kafka, watermark=[-(CASE(=(event_type, 0), person.dateTime, =(event_type, 1), auction.dateTime, bid.dateTime), 4000:INTERVAL SECOND)], watermarkEmitStrategy=[on-periodic]]], fields=[event_type, person, auction, bid])",
    "parallelism" : 4
  }, {
    "id" : 13,
    "type" : "MiniBatchAssigner[8]",
    "pact" : "Operator",
    "contents" : "[8]:MiniBatchAssigner(interval=[2000ms], mode=[ProcTime])",
    "parallelism" : 4,
    "predecessors" : [ {
      "id" : 12,
      "ship_strategy" : "FORWARD",
      "side" : "second"
    } ]
  }, {
    "id" : 14,
    "type" : "Calc[9]",
    "pact" : "Operator",
    "contents" : "[9]:Calc(select=[bid.channel AS channel, DATE_FORMAT(CAST(Reinterpret(CASE((event_type = 0), person.dateTime, (event_type = 1), auction.dateTime, bid.dateTime)) AS TIMESTAMP(3)), 'yyyy-MM-dd') AS day, DATE_FORMAT(CAST(Reinterpret(CASE((event_type = 0), person.dateTime, (event_type = 1), auction.dateTime, bid.dateTime)) AS TIMESTAMP(3)), 'HH:mm') AS $f2, (bid.price < 10000) IS TRUE AS $f3, SEARCH(bid.price, Sarg[[10000..1000000)]) IS TRUE AS $f4, (bid.price >= 1000000) IS TRUE AS $f5, bid.bidder AS bidder, bid.auction AS auction, MOD(HASH_CODE(DATE_FORMAT(CAST(Reinterpret(CASE((event_type = 0), person.dateTime, (event_type = 1), auction.dateTime, bid.dateTime)) AS TIMESTAMP(3)), 'HH:mm')), 1024) AS $f8, MOD(HASH_CODE(bid.bidder), 1024) AS $f9, MOD(HASH_CODE(bid.auction), 1024) AS $f10], where=[(event_type = 2)])",
    "parallelism" : 4,
    "predecessors" : [ {
      "id" : 13,
      "ship_strategy" : "FORWARD",
      "side" : "second"
    } ]
  }, {
    "id" : 15,
    "type" : "Expand[10]",
    "pact" : "Operator",
    "contents" : "[10]:Expand(projects=[{channel, day, $f2, $f3, $f4, $f5, bidder, auction, $f8, null AS $f9, null AS $f10, 3 AS $e}, {channel, day, $f2, $f3, $f4, $f5, bidder, auction, null AS $f8, $f9, null AS $f10, 5 AS $e}, {channel, day, $f2, $f3, $f4, $f5, bidder, auction, null AS $f8, null AS $f9, $f10, 6 AS $e}, {channel, day, $f2, $f3, $f4, $f5, bidder, auction, null AS $f8, null AS $f9, null AS $f10, 7 AS $e}])",
    "parallelism" : 4,
    "predecessors" : [ {
      "id" : 14,
      "ship_strategy" : "FORWARD",
      "side" : "second"
    } ]
  }, {
    "id" : 16,
    "type" : "Calc[11]",
    "pact" : "Operator",
    "contents" : "[11]:Calc(select=[channel, day, $f2, $f3, $f4, $f5, bidder, auction, $f8, $f9, $f10, ($e = 3) AS $g_3, ($e = 7) AS $g_7, (($e = 7) AND $f3) AS $g_70, (($e = 7) AND $f4) AS $g_71, (($e = 7) AND $f5) AS $g_72, ($e = 5) AS $g_5, (($e = 5) AND $f3) AS $g_50, (($e = 5) AND $f4) AS $g_51, (($e = 5) AND $f5) AS $g_52, ($e = 6) AS $g_6, (($e = 6) AND $f3) AS $g_60, (($e = 6) AND $f4) AS $g_61, (($e = 6) AND $f5) AS $g_62])",
    "parallelism" : 4,
    "predecessors" : [ {
      "id" : 15,
      "ship_strategy" : "FORWARD",
      "side" : "second"
    } ]
  }, {
    "id" : 17,
    "type" : "LocalGroupAggregate[12]",
    "pact" : "Operator",
    "contents" : "[12]:LocalGroupAggregate(groupBy=[channel, day, $f8, $f9, $f10], partialFinalType=[PARTIAL], select=[channel, day, $f8, $f9, $f10, MAX($f2) FILTER $g_3 AS max$0, COUNT(*) FILTER $g_7 AS count1$1, COUNT(*) FILTER $g_70 AS count1$2, COUNT(*) FILTER $g_71 AS count1$3, COUNT(*) FILTER $g_72 AS count1$4, COUNT(distinct$0 bidder) FILTER $g_5 AS count$5, COUNT(distinct$0 bidder) FILTER $g_50 AS count$6, COUNT(distinct$0 bidder) FILTER $g_51 AS count$7, COUNT(distinct$0 bidder) FILTER $g_52 AS count$8, COUNT(distinct$1 auction) FILTER $g_6 AS count$9, COUNT(distinct$1 auction) FILTER $g_60 AS count$10, COUNT(distinct$1 auction) FILTER $g_61 AS count$11, COUNT(distinct$1 auction) FILTER $g_62 AS count$12, DISTINCT(bidder) AS distinct$0, DISTINCT(auction) AS distinct$1])",
    "parallelism" : 4,
    "predecessors" : [ {
      "id" : 16,
      "ship_strategy" : "FORWARD",
      "side" : "second"
    } ]
  }, {
    "id" : 19,
    "type" : "IncrementalGroupAggregate[14]",
    "pact" : "Operator",
    "contents" : "[14]:IncrementalGroupAggregate(partialAggGrouping=[channel, day, $f8, $f9, $f10], finalAggGrouping=[channel, day], select=[channel, day, MAX(max$0) AS max$0, COUNT(count1$1) AS count1$1, COUNT(count1$2) AS count1$2, COUNT(count1$3) AS count1$3, COUNT(count1$4) AS count1$4, COUNT(distinct$0 count$5) AS count$5, COUNT(distinct$0 count$6) AS count$6, COUNT(distinct$0 count$7) AS count$7, COUNT(distinct$0 count$8) AS count$8, COUNT(distinct$1 count$9) AS count$9, COUNT(distinct$1 count$10) AS count$10, COUNT(distinct$1 count$11) AS count$11, COUNT(distinct$1 count$12) AS count$12])",
    "parallelism" : 4,
    "predecessors" : [ {
      "id" : 17,
      "ship_strategy" : "HASH",
      "side" : "second"
    } ]
  }, {
    "id" : 21,
    "type" : "GlobalGroupAggregate[16]",
    "pact" : "Operator",
    "contents" : "[16]:GlobalGroupAggregate(groupBy=[channel, day], partialFinalType=[FINAL], select=[channel, day, MAX(max$0) AS $f2, $SUM0(count1$1) AS $f3, $SUM0(count1$2) AS $f4, $SUM0(count1$3) AS $f5, $SUM0(count1$4) AS $f6, $SUM0(count$5) AS $f7, $SUM0(count$6) AS $f8, $SUM0(count$7) AS $f9, $SUM0(count$8) AS $f10, $SUM0(count$9) AS $f11, $SUM0(count$10) AS $f12, $SUM0(count$11) AS $f13, $SUM0(count$12) AS $f14])",
    "parallelism" : 4,
    "predecessors" : [ {
      "id" : 19,
      "ship_strategy" : "HASH",
      "side" : "second"
    } ]
  }, {
    "id" : 25,
    "type" : "nexmark_q16[17]: Writer",
    "pact" : "Operator",
    "contents" : "nexmark_q16[17]: Writer",
    "parallelism" : 4,
    "predecessors" : [ {
      "id" : 21,
      "ship_strategy" : "FORWARD",
      "side" : "second"
    } ]
  } ]
}"""

    def test_sql_plan(self):
        plan_dict = json.loads(self.plan)
        h = generate_plan_hash(plan_dict)
        print("Hash: " + h)
