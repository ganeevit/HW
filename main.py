import sys
import os
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, Window

# Get the input parameters
input_path1 = sys.argv[1]

# Create a SparkSession
spark = SparkSession.builder.appName("crime_analysis").getOrCreate()

# Read the CSV files
crimes = spark.read.option("header", "true").option("delimiter", ",").option("quote", "\"").option("escape", "\"").option("inferSchema", "true").csv(os.path.join(input_path1, "crime.csv")).filter(col("DISTRICT").isNotNull()).distinct()
codes = spark.read.option("header", "true").option("delimiter", ",").option("quote", "\"").option("escape", "\"").option("inferSchema", "true").csv(os.path.join(input_path1, "offense_codes.csv")).groupBy("CODE").agg(max("NAME")).distinct()

# Temporary tables
part1_count = crimes.groupBy("DISTRICT", "YEAR", "MONTH").agg(count("INCIDENT_NUMBER").alias("CRIMES_COUNT")).groupBy("DISTRICT").agg(sum("CRIMES_COUNT").alias("CRIMES_TOTAL"), expr("percentile_approx(CRIMES_COUNT, 0.5)").alias("CRIMES_MONTHLY"))
part2_latlong = crimes.groupBy("DISTRICT").agg(avg("lat").alias("LAT"), avg("long").alias("LNG"))

# Join the dataframes
joined_crimes_codes = crimes.join(codes, crimes["OFFENSE_CODE"] == codes["CODE"], "left")
window = Window.partitionBy("DISTRICT").orderBy(desc("CRIMES_TOTAL"))
part3_count = joined_crimes_codes.withColumn("NAME", split(col("NAME"), "-").getItem(0)).groupBy("DISTRICT", "NAME").agg(count("INCIDENT_NUMBER").alias("CRIMES_TOTAL"))
part3_top3 = part3_count.withColumn("rn", row_number().over(window)).filter(col("rn") <= 3).groupBy("DISTRICT").agg(collect_list("NAME").alias("top3_names")).select("DISTRICT", concat_ws(",", "top3_names").alias("FREQUENT_CRIMES_TYPES"))

# Join the dataframes
join_part1_part2 = part1_count.join(part2_latlong, part1_count["DISTRICT"] == part2_latlong["DISTRICT"], "left").drop(part2_latlong["DISTRICT"])
join_part1_part2_part3 = join_part1_part2.join(part3_top3, join_part1_part2["DISTRICT"] == part3_top3["DISTRICT"], "left").drop(part3_top3["DISTRICT"]).orderBy("DISTRICT")

# Write the result to a Parquet file
output_path = sys.argv[2]
join_part1_part2_part3.write.mode('overwrite').parquet(output_path)