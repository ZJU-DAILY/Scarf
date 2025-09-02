HDFS_HOSTS="your hosts"

mvn spotless:apply && mvn package && for host in $HDFS_HOSTS; do
  hdfs dfs -put -f target/flink-tune-job-0.1.jar hdfs://$host:9000/flink/jobs && echo "Uploaded to hdfs://$host:9000"
  scp target/flink-tune-job-0.1.jar $host:/home/User/code/stream-tuning/flink-tune-job/target/flink-tune-job-0.1.jar && echo "Copied to $host"
done
