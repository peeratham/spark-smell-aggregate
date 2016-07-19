"""Analyze result in MongoDB and output to latex"""

## Imports
from __future__ import division
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import pymongo_spark
from pyspark.sql.functions import udf
from pyspark.sql.types import *


## Constants
APP_NAME = "Large-Scale Block Smell Analysis"

# 'mongodb://hslogin1:27017'
##OTHER FUNCTIONS/CLASSES



## if __name__ == "__main__":
# Activate pymongo
pymongo_spark.activate()
# Configure Spark
conf = SparkConf().setAppName(APP_NAME)
sc   = SparkContext(conf=conf)
# Configure SQLContext
sqlContext = SQLContext(sc)
dbhost = sys.argv[1]
dbname = sys.argv[2]
format_dir = sys.argv[3]
output_dir = sys.argv[4]

def write_latex(pdf, filename):
    with open(filename, 'w') as f:
        f.truncate()
        f.write(pdf.to_latex())

#def main(sc, sqlContext, dbname):
# get reports
# report_sample_df = sqlContext.read.load("/home/tpeera4/projects/scripts/spark-mongo-analysis/report-format.json", format="json")
report_sample_df = sqlContext.read.load(format_dir+"reports.json", format="json")
reports_rdd = sc.mongoRDD(dbhost+dbname+'.reports')
reports_df = sqlContext.createDataFrame(reports_rdd, report_sample_df.schema)

#smell frequency
smell_freq_df = reports_df.select([cname+'.count' if cname != '_id' else cname for cname in reports_df.columns])
smell_freq_df = smell_freq_df.toDF(*reports_df.columns)


# get metadata
project_metadata_rdd = sc.mongoRDD(dbhost+dbname+'.metadata')
project_metadata_df = sqlContext.createDataFrame(project_metadata_rdd)
# filter original project
project_metadata_df = project_metadata_df.filter(project_metadata_df['_id'] == project_metadata_df['original'])

# get creators
creators_rdd = sc.mongoRDD(dbhost+dbname+'.creators')
creators_df = sqlContext.createDataFrame(creators_rdd)
creators_df = creators_df.select(creators_df['_id'].alias('creator'),creators_df['mastery.total'])
# get metrics
metrics_sample_df = sqlContext.read.load(format_dir+"metrics.json", format="json")
metrics_rdd = sc.mongoRDD(dbhost+dbname+'.metrics')
metrics_df = sqlContext.createDataFrame(metrics_rdd, metrics_sample_df.schema)
# filter trivial projects (zero scripts, 0-1 sprite)?
metrics_df = metrics_df.filter(metrics_df['PS.script']>1)

 ## join with reports_df then count total
analysis_df = project_metadata_df.join(smell_freq_df, '_id')
analysis_df = analysis_df.join(metrics_df, '_id')
analysis_df = analysis_df.join(creators_df, 'creator')
smells = [cname for cname in reports_df.columns if cname != '_id']

#################Average Smells Per Script#######################################
smell_stats_pdf = analysis_df.select(smells).describe().toPandas().transpose()
smell_stats_pdf.columns = smell_stats_pdf.iloc[0]
smell_stats_pdf = smell_stats_pdf.reindex(smell_stats_pdf.index.drop('summary'))
smell_stats_pdf = smell_stats_pdf.drop('count',1)
smell_stats_pdf = smell_stats_pdf.drop('min',1)
smell_stats_pdf = smell_stats_pdf.apply(lambda x: pd.to_numeric(x, errors='coerce'))
write_latex(smell_stats_pdf ,output_dir+'smell_stats.tex')




#################Average Smells Per Script#######################################
get_count = lambda record: record['count'] if isinstance(record, dict) else record
udf_get_count = udf(get_count, IntegerType())

smell_freq_df = reports_df.select([cname+'.count' if cname != '_id' else cname for cname in reports_df.columns])
smell_freq_df = smell_freq_df.toDF(*reports_df.columns)

# filter trivial projects (zero scripts, 0-1 sprite)?
metric_criteria_df = metrics_df.filter(metrics_df['PS.script']>1)
smell_metric_df = smell_freq_df.join(metric_criteria_df, '_id')

# normalized (divided by scriptCounts to account for various project size)
normalized_by_script = lambda smell, script: smell/script
udf_norm_smell = udf(normalized_by_script, DoubleType())
norm_smell_metric_df = smell_metric_df.select([udf_norm_smell(smell_metric_df[smell], smell_metric_df['PS.script']).alias(smell) for smell in smells])

average_smells_per_script = norm_smell_metric_df.groupby().avg(*smells).toDF(*smells)
average_smells_per_script_pdf = average_smells_per_script.toPandas()
average_smells_per_script_pdf = average_smells_per_script_pdf.transpose()
average_smells_per_script_pdf.columns=['Avg Smells per Script']
write_latex(average_smells_per_script_pdf , output_dir+'smell_per_script.tex')
#################################################################################
################Percent of each smells in the entire population##################
exists = lambda col: 1 if col > 0 else 0
udf_exists = udf(exists, IntegerType())
distinct_smell_df = smell_metric_df.select(*[udf_exists(column).alias(column) for column in smells])
with_total_distinct_smells_df = distinct_smell_df.withColumn('Distinct Smell Counts', sum([distinct_smell_df[smell] for smell in smells]))
row_counts = with_total_distinct_smells_df.count()
found_smell_sum = with_total_distinct_smells_df.groupby().sum(*smells).toDF(*smells)
found_smell_sum_pdf = found_smell_sum.toPandas()
percentage_smell_pdf = found_smell_sum_pdf.applymap(lambda found: found/row_counts*100)
percentage_smell_pdf = percentage_smell_pdf.transpose()
percentage_smell_pdf.columns=['freq (%)']
write_latex(percentage_smell_pdf ,output_dir+'percent_smell_found.tex')
##################################################################################
#############################Distinct Smells######################################
################Summary Smell Stats##################
combined_stats_pdf = average_smells_per_script_pdf.join(percentage_smell_pdf).round(2)
combined_stats_pdf.columns.name='Smell'
write_latex(combined_stats_pdf, output_dir+'smell_stats.tex')
#####################################################
###################Comparison########################
current_palette = sns.color_palette("husl", 10)
metrics = ['PS.script', 'PS.sprite','PS.bloc', 'PS.script_length']
metrics_pdf = analysis_df.groupBy('MS.total').avg(*metrics).toDF('Mastery Level', *metrics).toPandas()
metrics_pdf = metrics_pdf.set_index('Mastery Level')
metrics_pdf = metrics_pdf.plot(kind='bar', color=current_palette)
plt.savefig(output_dir+'mastery-metrics')

#smell per bloc (block line of code)
normalize = lambda counts, size: counts/size
udf_normalize = udf(normalize, DoubleType())
norm_analysis_df = analysis_df.select(*[udf_normalize(column, analysis_df['PS.bloc']).alias(column) if column in smells else column for column in analysis_df.columns])

smells_pdf = norm_analysis_df.groupBy('MS.total').avg(*smells).toDF('Mastery Level',*smells).toPandas()
smells_pdf = smells_pdf.set_index('Mastery Level')
smells_pdf.plot(kind='bar', stacked=True, color=current_palette)
plt.savefig(output_dir+'mastery-smell')

#population distribution
level_dist_pdf = analysis_df.groupBy('MS.total').count().toDF('Mastery Level', 'count').toPandas()
level_dist_pdf = level_dist_pdf.set_index('Mastery Level')
level_dist_pdf.plot(kind='bar', stacked=True, color=current_palette)
plt.savefig(output_dir+'mastery-distribution')

#Basic Stats
# distinct_creators = analysis_df.select('creator').distinct().count()
# analysis_population = smell_stats_pdf[0][1]

