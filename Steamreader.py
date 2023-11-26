from delta import *
from delta.tables import *
import pyspark
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType, DoubleType, BooleanType
from pyspark.sql.functions import col, expr, monotonically_increasing_id
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import findspark
import re 
import sparknlp
from pyspark.ml.feature import Tokenizer
from sparknlp.annotator import SentimentDetector
from sparknlp.base import *
findspark.init()
import sparknlp
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.base import *
from sparknlp.annotator import *

builder = pyspark.sql.SparkSession.builder.appName("SteamReaderTestFile") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.executor.instances", 3) \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
    .config("spark.kryoserializer.buffer.max", "200M")\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_5.1.4:3.5.0")\
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
        (col("columns")[0].cast(IntegerType()).isNotNull()) & #index
        (col("columns")[1].cast(IntegerType()).isNotNull()) & #app_id
        (col("columns")[2].cast(StringType()).isNotNull()) & #app_name
        (col("columns")[3].cast(IntegerType()).isNotNull()) & #review_id
    #    (col("columns")[4].cast(StringType()).isNotNull()) & #language
        (col("columns")[5].cast(StringType()).isNotNull()) & #review
    #    (col("columns")[6].cast(IntegerType()).isNotNull()) & #timestamp_created
    #    (col("columns")[7].cast(IntegerType()).isNotNull()) & #timestamp_updated
        (col("columns")[8].cast(BooleanType()).isNotNull()) & #Recommended
        (col("columns")[9].cast(IntegerType()).isNotNull()) & #votes_helpful
    #    (col("columns")[10].cast(IntegerType()).isNotNull()) & #votes_funny
    #    (col("columns")[11].cast(DoubleType()).isNotNull()) & #weighted_vote_score
        (col("columns")[12].cast(IntegerType()).isNotNull()) & #commentcount
        (col("columns")[13].cast(BooleanType()).isNotNull()) & #steam_purchase
    #    (col("columns")[14].cast(BooleanType()).isNotNull()) & #received_for_free
    #    (col("columns")[15].cast(BooleanType()).isNotNull()) & #written_durin early access
    #    (col("columns")[16].cast(IntegerType()).isNotNull()) & #author_steamid
    #    (col("columns")[17].cast(IntegerType()).isNotNull()) & #author_numGames
    #    (col("columns")[18].cast(IntegerType()).isNotNull()) & #Author_num_reviews
        (col("columns")[19].cast(DoubleType()).isNotNull())  #Author_playtime
    #    (col("columns")[20].cast(DoubleType()).isNotNull()) & #author_playtime_last_two_weeks
    #    (col("columns")[21].cast(DoubleType()).isNotNull()) & #Author_playtime_at_review
    #    (col("columns")[22].cast(DoubleType()).isNotNull()) #Author_last_played
        
    )
    .selectExpr(
        "columns[0] as index",
        "columns[1] as app_id",
        "columns[2] as app_name",
        "columns[3] as review_id",
        "columns[4] as language",
        "columns[5] as review",
    #    "columns[6] as timestamp_created",
    #    "columns[7] as timestamp_updated",
        "columns[8] as recommended",
        "columns[9] as votes_helpful",
    #    "columns[10] as votes_funny",
    #    "columns[11] as weighted_vote_score",
        "columns[12] as comment_count",
        "columns[13] as steam_purchase",
        # "columns[14] as received_for_free",
        # "columns[15] as written_during_early_access",
        # "columns[16] as authorSteamid",
        # "columns[17] as authorNum_games_owned",
        # "columns[18] as authorNum_reviews",
        "columns[19] as authorPlaytime_forever",
        # "columns[20] as authorPlaytime_last_two_weeks",
        # "columns[21] as authorPlaytime_at_review",
        # "columns[22] as authorLast_played"
    )
)

# df = (
#     spark.read.text(hlist)
#     .selectExpr("split(value, ',') as columns")
#     .filter(
#         (col("columns")[0].cast(IntegerType()).isNotNull()) & #index #
#         (col("columns")[1].cast(IntegerType()).isNotNull()) & #app_id
#         (col("columns")[2].cast(StringType()).isNotNull()) & #app_name
#         (col("columns")[3].cast(IntegerType()).isNotNull()) & #review_id
#         (col("columns")[4].cast(StringType()).isNotNull()) & #language #
#         (col("columns")[5].cast(StringType()).isNotNull()) & #review
#         (col("columns")[6].cast(IntegerType()).isNotNull()) & #timestamp_created #
#         (col("columns")[7].cast(IntegerType()).isNotNull()) & #timestamp_updated #
#         (col("columns")[8].cast(BooleanType()).isNotNull()) & #Recommended
#         (col("columns")[9].cast(IntegerType()).isNotNull()) & #votes_helpful
#         (col("columns")[10].cast(IntegerType()).isNotNull()) & #votes_funny #
#         (col("columns")[11].cast(DoubleType()).isNotNull()) & #weighted_vote_score #
#         (col("columns")[12].cast(StringType()).isNotNull()) & #commentcount
#         (col("columns")[13].cast(BooleanType()).isNotNull()) & #steam_purchase
#         (col("columns")[14].cast(BooleanType()).isNotNull()) & #received_for_free #
#         (col("columns")[15].cast(BooleanType()).isNotNull()) & #written_durin early access #
#         (col("columns")[16].cast(StringType()).isNotNull()) & #author_steamid #
#         (col("columns")[17].cast(IntegerType()).isNotNull()) & #author_numGames #
#         (col("columns")[18].cast(IntegerType()).isNotNull()) & #Author_num_reviews #
#         (col("columns")[19].cast(DoubleType()).isNotNull())  & #Author_playtime 
#         (col("columns")[20].cast(DoubleType()).isNotNull()) & #author_playtime_last_two_weeks #
#         (col("columns")[21].cast(DoubleType()).isNotNull()) & #Author_playtime_at_review#
#         (col("columns")[22].cast(DoubleType()).isNotNull()) #Author_last_played#
        
#     )
#     .selectExpr(
#         "columns[0] as index",
#         "columns[1] as app_id",
#         "columns[2] as app_name",
#         "columns[3] as review_id",
#         "columns[4] as language",
#         "columns[5] as review",
#         "columns[6] as timestamp_created",
#         "columns[7] as timestamp_updated",
#         "columns[8] as recommended",
#         "columns[9] as votes_helpful",
#         "columns[10] as votes_funny",
#         "columns[11] as weighted_vote_score",
#         "columns[12] as comment_count",
#         "columns[13] as steam_purchase",
#         "columns[14] as received_for_free",
#         "columns[15] as written_during_early_access",
#         "columns[16] as authorSteamid",
#         "columns[17] as authorNum_games_owned",
#         "columns[18] as authorNum_reviews",
#         "columns[19] as authorPlaytime_forever",
#         "columns[20] as authorPlaytime_last_two_weeks",
#         "columns[21] as authorPlaytime_at_review",
#         "columns[22] as authorLast_played"
#     )
# )
#df.show()


df.createOrReplaceTempView("final_temp_view")


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

#percentage_df.cache()
percentage_df.show()

df = df.join(percentage_df, on='app_id', how='left')

#df.persist()

sparknlp.start()
df = df.dropna().withColumnRenamed("review", "text")
sentiment = PretrainedPipeline("analyze_sentiment", lang="en")
df = sentiment.transform(df)
df = df.withColumnRenamed("text", "review")
df = df.withColumn("predicted_sentiment", col("sentiment.result").getItem(0))
df = df.drop(*["document", "sentence", "token", "checked", "sentiment"])
df.show()
 
#df.unpersist()

#final_delta_table = "/steamReaderWithPercentages"

#Dette blir den nye verdien 
final_df_delta = df.write.format("delta")



if DeltaTable.isDeltaTable(spark, '/fullsteamReaderWithPercentagefilter'):
    print("Deltatable found")
    DeltaSteamDfWithPercentage = DeltaTable.forPath(spark, '/fullsteamReaderWithPercentagefilter')
    #dataFrameUpdates = DeltaSteamDfWithPercentageUpdate.toDF()
    #final_df_delta.alias('updates')

    DeltaSteamDfWithPercentage.alias('old') \
    .merge(
        df.alias('updates'),
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
    final_delta_table = final_df_delta.save('/fullsteamReaderWithPercentagefilter')
    final_delta_table = DeltaTable.forPath(spark, '/fullsteamReaderWithPercentagefilter')

# Save the DataFrame as a CSV file
#final_df.write.csv("SteamReviews.csv", header=True, mode="overwrite")

spark.stop()