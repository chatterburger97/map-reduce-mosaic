#! /bin/bash
wget -O calvin.jpg https://i.pinimg.com/originals/55/2a/7c/552a7cfa112565e94ff9cea42ee11abd.jpg
hdfs dfs -put calvin.jpg /user/bitnami/input/
