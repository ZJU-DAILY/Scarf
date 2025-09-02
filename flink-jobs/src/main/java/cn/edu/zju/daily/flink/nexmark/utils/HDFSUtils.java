package cn.edu.zju.daily.flink.nexmark.utils;

import java.net.URI;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class HDFSUtils {

    public static void write(String content, String path) {
        Pattern pattern = Pattern.compile("(hdfs://.*?)(/.*)");
        Matcher matcher = pattern.matcher(path);

        if (!matcher.matches()) {
            throw new IllegalArgumentException("Invalid HDFS path: " + path);
        }

        // Get matching groups
        String hdfsUrl = matcher.group(1); // hdfs://host:port
        String filePath = matcher.group(2); // /path/to/file

        // path: hdfs://host:port/path/to/file
        Configuration conf = new Configuration();
        conf.set("fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem");
        try (FileSystem fs = FileSystem.get(new URI(hdfsUrl), conf)) {
            Path hdfsPath = new Path(filePath);
            try (FSDataOutputStream outputStream = fs.create(hdfsPath, true)) {
                outputStream.writeBytes(content);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to write to HDFS path: " + path, e);
        }
    }

    public static void main(String[] args) {
        String content = "Hello, HDFS!";
        String hdfsPath =
                "hdfs://node181:9000/flink/job-plans/test.txt"; // Replace with your HDFS path
        write(content, hdfsPath);
        System.out.println("Content written to HDFS: " + hdfsPath);
    }
}
