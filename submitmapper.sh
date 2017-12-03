#! /bin/bash
spark-submit first_mapper.py --master spark://linux:7077 --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=hdfs://127.0.0.1:50010:8020/eventLogging
