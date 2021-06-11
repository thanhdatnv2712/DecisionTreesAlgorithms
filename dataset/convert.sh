# !/bin/bash
java -cp /home/datnvt/ssd/weka-3-8-5-azul-zulu-linux/weka-3-8-5/weka.jar weka.core.converters.CSVLoader $1_$2.csv > $1_$2.arff