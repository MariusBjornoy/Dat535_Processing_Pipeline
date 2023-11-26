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
import sparknlp
from pyspark.ml.feature import Tokenizer
from sparknlp.annotator import SentimentDetector
from sparknlp.base import *
findspark.init()

builder = pyspark.sql.SparkSession.builder.appName("SteamReader") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.executor.instances", 3) \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
    .config("spark.kryoserializer.buffer.max", "200M")\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_5.1.4:3.5.0")\
    .config("spark.executor.cores", 4) 
#    .config("spark.executor.memory", "1g") 


# Create a SparkSession
#spark = SparkSession.builder.appName("steamreader").getOrCreate()
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Read the text file with the given delimiter
# hlist = "/ProjectDat535/steam_reviews.txt"
hlist = "/ProjectDat535/steam_reviews.txt"

schema = StructType([
#    StructField("index", IntegerType(), True), #index
    StructField("app_id", IntegerType(), True), #app_Id
    StructField("app_name", StringType(), True), #app_name
    StructField("review_id", IntegerType(), True), #review_id
#    StructField("language", StringType(), True), #language
    StructField("review", StringType(), True), #review
#    StructField("timestamp_created", LongType(), True), #Timestamp created
#    StructField("timestamp_updated", LongType(), True), #timestamp updated
    StructField("recommended", BooleanType(), True), #recommended
    StructField("votes_helpful", IntegerType(), True), #votes helpful 
#    StructField("votes_funny", IntegerType(), True), #votes funny
#    StructField("weighted_vote_score", DoubleType(), True), #weighted vote score
    StructField("comment_count", IntegerType(), True), #comment count
    StructField("steam_purchase", BooleanType(), True), #steam purchase
    # StructField("received_for_free", BooleanType(), True), #received for free
    # StructField("written_during_early_access", BooleanType(), True), # written during early access
    # StructField("author.steamid", IntegerType(), True), #author_steam id
    # StructField("author.num_games_owned", IntegerType(), True), #author number of games
    # StructField("author.num_reviews", IntegerType(), True), #author num reviews
    StructField("author.playtime_forever", DoubleType(), True), #Author playtime
    # StructField("author.playtime_last_two_weeks", DoubleType(), True), #Author playtime last two weeks
    # StructField("author.playtime_at_review", DoubleType(), True), #author playtime at review
    # StructField("author.last_played", DoubleType(), True) #author last played
])

#Upsert



df = (
    spark.read.text(hlist)
    .selectExpr("split(value, ',') as columns")
    .filter(
    #    (col("columns")[0].cast(IntegerType()).isNotNull()) & #index
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
    #    "columns[0] as index",
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


##############################################################################
###################### Dette skal være med Marius ############################
##############################################################################
import sparknlp
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.base import *
from sparknlp.annotator import *

sparknlp.start()

df = df.dropna().withColumnRenamed("review", "text")

sentiment = PretrainedPipeline("analyze_sentiment", lang="en")
df = sentiment.transform(df)
df = df.withColumnRenamed("text", "review")
df = df.withColumn("predicted_sentiment", col("sentiment.result").getItem(0))
df = df.drop(*["document", "sentence", "token", "checked", "sentiment"])
df.show()

spark.stop()