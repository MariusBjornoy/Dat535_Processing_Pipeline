from delta import *
from delta.tables import *
import pyspark
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType, DoubleType, BooleanType
from pyspark.sql.functions import col, expr, udf
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import findspark
import re 
from nltk.corpus import stopwords
findspark.init()

builder = pyspark.sql.SparkSession.builder.appName("SteamReader") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.executor.instances", 3) \
    .config("spark.executor.cores", 4) 
#    .config("spark.executor.memory", "1g") 


spark = configure_spark_with_delta_pip(builder).getOrCreate()

hlist = "/ProjectDat535/steam_reviews.txt"
#Abriviated
#hlist = "/ProjectDat535/abrieiatedSteamreview.txt"

df = (
    spark.read.text(hlist)
    .selectExpr("split(value, ',') as columns")
    .filter(
        (col("columns")[0].cast(IntegerType()).isNotNull()) & #index #
        (col("columns")[1].cast(IntegerType()).isNotNull()) & #app_id
        (col("columns")[2].cast(StringType()).isNotNull()) & #app_name
        (col("columns")[3].cast(IntegerType()).isNotNull()) & #review_id
        (col("columns")[4].cast(StringType()).isNotNull()) & #language #
        (col("columns")[5].cast(StringType()).isNotNull()) & #review
        (col("columns")[6].cast(IntegerType()).isNotNull()) & #timestamp_created #
        (col("columns")[7].cast(IntegerType()).isNotNull()) & #timestamp_updated #
        (col("columns")[8].cast(BooleanType()).isNotNull()) & #Recommended
        (col("columns")[9].cast(IntegerType()).isNotNull()) & #votes_helpful
        (col("columns")[10].cast(IntegerType()).isNotNull()) & #votes_funny #
        (col("columns")[11].cast(DoubleType()).isNotNull()) & #weighted_vote_score #
        (col("columns")[12].cast(StringType()).isNotNull()) & #commentcount
        (col("columns")[13].cast(BooleanType()).isNotNull()) & #steam_purchase
        (col("columns")[14].cast(BooleanType()).isNotNull()) & #received_for_free #
        (col("columns")[15].cast(BooleanType()).isNotNull()) & #written_durin early access #
        (col("columns")[16].cast(StringType()).isNotNull()) & #author_steamid #
        (col("columns")[17].cast(IntegerType()).isNotNull()) & #author_numGames #
        (col("columns")[18].cast(IntegerType()).isNotNull()) & #Author_num_reviews #
        (col("columns")[19].cast(DoubleType()).isNotNull())  & #Author_playtime 
        (col("columns")[20].cast(DoubleType()).isNotNull()) & #author_playtime_last_two_weeks #
        (col("columns")[21].cast(DoubleType()).isNotNull()) & #Author_playtime_at_review#
        (col("columns")[22].cast(DoubleType()).isNotNull()) #Author_last_played#
        
    )
    .selectExpr(
        "columns[0] as index",
        "columns[1] as app_id",
        "columns[2] as app_name",
        "columns[3] as review_id",
        "columns[4] as language",
        "columns[5] as review",
        "columns[6] as timestamp_created",
        "columns[7] as timestamp_updated",
        "columns[8] as recommended",
        "columns[9] as votes_helpful",
        "columns[10] as votes_funny",
        "columns[11] as weighted_vote_score",
        "columns[12] as comment_count",
        "columns[13] as steam_purchase",
        "columns[14] as received_for_free",
        "columns[15] as written_during_early_access",
        "columns[16] as authorSteamid",
        "columns[17] as authorNum_games_owned",
        "columns[18] as authorNum_reviews",
        "columns[19] as authorPlaytime_forever",
        "columns[20] as authorPlaytime_last_two_weeks",
        "columns[21] as authorPlaytime_at_review",
        "columns[22] as authorLast_played"
    )
)
#df.show()

stopwords_df = spark.read.text("/ProjectDat535/stopwords.txt")
stopwords_list = stopwords_df.rdd.map(lambda row: row[0].strip()).collect()



# Read stopwords from file into a Spark DataFrame
stopwords_df = spark.read.text("/ProjectDat535/stopwords.txt")
stopwords_list = stopwords_df.rdd.map(lambda row: row[0].strip()).collect()

# Define the cleaning function
def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'(http\S+)|(www\S+)|(\S+.com)', ' ', text)  # Remove URLs
    text = re.sub(r'[^\w\s\u4E00-\u9FFF]', '', text)

    text = ' '.join([word for word in text.split() if word not in stopwords_list])


df.createOrReplaceTempView("final_temp_view")


df_ml, _ = df.randomSplit([0.1, 0.9], seed=123)


clean_text_udf = udf(lambda text: clean_text(text), StringType())
df_ml = df_ml.withColumn("review", clean_text_udf(df["review"]))

# ml_df = df.select("review", "recommended")
indexer = StringIndexer(inputCol="recommended", outputCol="label")
df_ml = indexer.fit(df_ml).transform(df_ml)

false_count = df_ml.filter(df_ml["recommended"] == "False").count()

# Display the count
print("Number of times 'recommended' is False: {}".format(false_count))

tokenizer = Tokenizer(inputCol="review", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
nb = NaiveBayes(labelCol="label", featuresCol="features", predictionCol="prediction", smoothing=1.0)
# pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb])
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb])

train_data, test_data = df_ml.randomSplit([0.8, 0.2], seed=123)
model = pipeline.fit(train_data)
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy: {:.2%}".format(accuracy))


#result = spark.sql("SELECT app_id, COLLECT_LIST(recommended) AS recommended_values FROM temp_view_name GROUP BY app_id")
#result.show()

recommended_counts = spark.sql("""
    SELECT app_id, 
           SUM(CASE WHEN recommended = TRUE THEN 1 ELSE 0 END) AS recommended_count,
           SUM(CASE WHEN recommended = FALSE THEN 1 ELSE 0 END) AS not_recommended_count
    FROM final_temp_view
    GROUP BY app_id
""")
#recommended_counts.show(10)

percentage_df = recommended_counts.withColumn(
    "percentage_liked",
    expr("recommended_count / (recommended_count + not_recommended_count) * 100")
)

#percentage_df.cache
percentage_df.show()

final_df_with_percentage = df.join(percentage_df, on='app_id', how='left')









#brodcast join
#final_df_with_percentage = df.join(broadcast(percentage_df), on='app_id', how='left')


#percentage_df.unpersist()
#Creating a delta table from 
final_delta_table = "/steamReaderWithPercentages"

#Dette blir den nye verdien 
final_df_delta = final_df_with_percentage.write.format("delta")


if DeltaTable.isDeltaTable(spark, '/fullsteamReaderWithPercentage'):
    print("Deltatable found")
    DeltaSteamDfWithPercentage = DeltaTable.forPath(spark, '/fullsteamReaderWithPercentage')
    #dataFrameUpdates = DeltaSteamDfWithPercentageUpdate.toDF()
    #final_df_delta.alias('updates')

    DeltaSteamDfWithPercentage.alias('old') \
    .merge(
        final_df_with_percentage.alias('updates'),
        'old.index = updates.index'
    ).whenMatchedUpdate(set= 
                        {
                            "app_id": "updates.app_id",
                            "app_name": "updates.app_name",
                            "review_id": "updates.review_id",
                            "review": "updates.review",
                            "recommended": "updates.recommended",
                            "votes_helpful": "updates.votes_helpful",
                            "comment_count": "updates.comment_count",
                            "steam_purchase": "updates.steam_purchase",
                            "authorPlaytime_forever": "updates.authorPlaytime_forever",
                            "recommended_count": "updates.recommended_count",
                            "not_recommended_count": "updates.not_recommended_count",
                            "percentage_liked": "updates.percentage_liked"
                        }).whenNotMatchedInsert(values = 
                        {
                            "app_id": "updates.app_id",
                            "app_name": "updates.app_name",
                            "review_id": "updates.review_id",
                            "review": "updates.review",
                            "recommended": "updates.recommended",
                            "votes_helpful": "updates.votes_helpful",
                            "comment_count": "updates.comment_count",
                            "steam_purchase": "updates.steam_purchase",
                            "authorPlaytime_forever": "updates.authorPlaytime_forever",
                            "recommended_count": "updates.recommended_count",
                            "not_recommended_count": "updates.not_recommended_count",
                            "percentage_liked": "updates.percentage_liked"
                        }
                        ).execute()
    print("Table was updated")
else:
    print("no table found, saving it now")
    final_delta_table = final_df_delta.save('/fullsteamReaderWithPercentage')
    final_delta_table = DeltaTable.forPath(spark, '/fullsteamReaderWithPercentage')

# Save the DataFrame as a CSV file
#final_df.write.csv("SteamReviews.csv", header=True, mode="overwrite")

spark.stop()