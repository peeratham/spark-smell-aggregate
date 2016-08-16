##Analyze result in MongoDB and output to latex

## Imports
from __future__ import division
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import matplotlib
matplotlib.use('Agg') #needs to set before other import to work properly

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate

import pymongo_spark
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql.functions import mean, min, max

import numpy as np
import math

## Constants
APP_NAME = "Large-Scale Block Smell Analysis"

# Activate pymongo
pymongo_spark.activate()
# Configure Spark
conf = SparkConf().setAppName(APP_NAME)

local = False
# Configure SQLContext
try:
   sc = SparkContext(conf=conf)
   dbhost = 'mongodb://'+sys.argv[1]+':27017/'
   dbname = sys.argv[2]
   format_dir = sys.argv[3]
   output_dir = sys.argv[4]
except:
   print('run in shell mode')
   dbname = 'analysis'
   if local :
      dbhost = 'mongodb://localhost:27017/'
      format_dir = '/home/peeratham/tpeera4/smell-analysis/format/'
      output_dir = '/home/peeratham/tpeera4/smell-analysis/analysis_output/'
   else:
      dbhost = 'mongodb://hs082:27017/'
      format_dir = 'file:///home/tpeera4/projects/spark-smell-aggregate/'
      output_dir = '/home/tpeera4/analysis_output/'
      #output_dir = '/home/tpeera4/analysis_output/tex/'

sqlContext = SQLContext(sc)

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def write_latex(pdf, filename):
    with open(filename, 'w') as f:
        f.truncate()
        f.write(pdf.to_latex())

def write_to_file(string, filename):
    with open(filename, 'w') as f:
        f.truncate()
        f.write(string)

# get reports
# local: sqlContext.read.load("/home/tpeera4/projects/scripts/spark-mongo-analysis/report-format.json", format="json")
report_sample_df = sqlContext.read.load(format_dir+"reports.json", format="json")
reports_rdd = sc.mongoRDD(dbhost+dbname+'.reports')
reports_df = sqlContext.createDataFrame(reports_rdd, report_sample_df.schema)
## get all smell names
smells = [cname for cname in reports_df.columns if cname != '_id']

#smell frequency
# smell_freq_df = reports_df.select([cname if cname != '_id' else cname for cname in reports_df.columns])
# smell_freq_df = smell_freq_df.toDF(*reports_df.columns)
# get metadata
project_metadata_rdd = sc.mongoRDD(dbhost+dbname+'.metadata')
project_metadata_df = sqlContext.createDataFrame(project_metadata_rdd)

# get creators
creators_rdd = sc.mongoRDD(dbhost+dbname+'.creators')
creators_df = sqlContext.createDataFrame(creators_rdd)
creators_df = creators_df.select(creators_df['_id'].alias('creator'),creators_df['mastery.total'])

# get metrics
metrics_sample_df = sqlContext.read.load(format_dir+"metrics.json", format="json")
metrics_rdd = sc.mongoRDD(dbhost+dbname+'.metrics')
metrics_df = sqlContext.createDataFrame(metrics_rdd, metrics_sample_df.schema)

## join with reports_df then count total
# analysis_df = project_metadata_df.join(smell_freq_df, '_id')
analysis_df = project_metadata_df.join(metrics_df, '_id')
analysis_df = analysis_df.join(metrics_df, '_id')
analysis_df = analysis_df.join(creators_df, 'creator')

###############Derived Information###########################
#Mastery Level (1-3)# floor of score/7 (1:Basic, 2:Developing, 3:Proficiency). 
udf_mastery_level = udf(lambda score: int(math.ceil(float(score/7))), IntegerType())
analysis_df = analysis_df.withColumn('masteryLevel', udf_mastery_level(analysis_df['total']))
#Project Size Level#
def bloc_level(col):
   if col < 300: return 'small'
   elif col > 300 and col <1000: return 'medium'
   else: return 'large'

udf_bloc_level = udf(bloc_level, StringType())
analysis_df = analysis_df.withColumn('bloc level', udf_bloc_level(analysis_df['PS.bloc']))

##################Filtering################################
## filter trivial projects (zero scripts, 0-1 sprite)?
# original_analysis_df = analysis_df
zero_script_count = analysis_df.filter(metrics_df['PS.script']==0).count() 
analysis_df = analysis_df.filter(analysis_df['PS.script']>0)
analysis_df = analysis_df.filter(analysis_df['PS.bloc']>0)
analysis_df= analysis_df.filter(analysis_df['masteryLevel']>0)

# no var filter (UV, BVS)
no_var_project_count = analysis_df.filter(analysis_df['PE.varCount']==0).count()
#Specific Dataset Filtering#
var_smell_analysis_df= analysis_df.filter(analysis_df['PE.varCount']>0)

# no custom block filter (UCB)
no_custom_block_project_count = analysis_df.filter(analysis_df['PE.customBlockCount']==0).count()
#keep remix
no_remix_filter_analysis_df =  analysis_df
# filter out remix project
analysis_df = analysis_df.filter(analysis_df['_id'] == analysis_df['original'])

##Summing Smell
##cannot sum!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
original_analysis_df = original_analysis_df.withColumn('totalSmell', sum(original_analysis_df[smell+'.count'] for smell in smells))


##############normalization#####################
##smell per bloc (block line of code)
normalize = lambda counts, size: 0.0 if counts==0 else counts/size
udf_normalize = udf(normalize, DoubleType())

# var_related_smells = ['UV']
# custom_block_smells = ['UCB']
# non_loc_denom_smells = var_related_smells + custom_block_smells
norm_analysis_df = analysis_df.select(*[udf_normalize(column+'.count', analysis_df['PS.bloc']).alias(column) if column in smells else column for column in analysis_df.columns])
norm_original_analysis_df = original_analysis_df.select(*[udf_normalize(column+'.count', original_analysis_df['PS.bloc']).alias(column) if column in smells else column for column in original_analysis_df.columns])
norm_original_analysis_df = original_analysis_df.withColumn('totalSmell',udf_normalize(original_analysis_df['totalSmell'], original_analysis_df['PS.bloc']))

# norm_analysis_df = norm_analysis_df.select(*[udf_normalize(column+'.count', norm_analysis_df['PE.varCount']).alias(column) if column in var_related_smells else column for column in norm_analysis_df.columns])
# norm_analysis_df = norm_analysis_df.select(*[udf_normalize(column+'.count', norm_analysis_df['PE.customBlockCount']).alias(column) if column in custom_block_smells else column for column in norm_analysis_df.columns])






###################################################################
###################---Basic Stats---###############################
###################################################################
#Dataset Stats
projects_collected = project_metadata_df.count()
total_projects = no_remix_filter_analysis_df.count() 
total_creators = no_remix_filter_analysis_df.groupBy('creator').count().count()
## original work vs remix
original_count = analysis_df.count()
remix_project_metadata_df = no_remix_filter_analysis_df.filter(no_remix_filter_analysis_df['_id'] != no_remix_filter_analysis_df['original'])
remix_count = remix_project_metadata_df.count()

print "Number of projects collected:", projects_collected #1066308 
print "total_projects_analyzed:" , total_projects #886344
print "zero_script_count:", zero_script_count #70702
print "Percent of zero_script_count:", zero_script_count/total_projects*100 #6.82%
print "Number of creators:", total_creators #271913
print "original projects:" , original_count, "percent:", original_count/total_projects*100   #original projects: 672224 percent: 75.84
print "remix projects:" , remix_count, "percent:", remix_count/total_projects*100   #remix projects: 214120 percent: 24.16
print "Project with variables:", total_projects-no_var_project_count, (total_projects-no_var_project_count)/total_projects*100 #330887 , 37.56%
print "Project with custom block", total_projects-no_custom_block_project_count, (total_projects-no_custom_block_project_count)/total_projects*100 #78320 , 8.89%
#################################################################
#######################-DATASET DISTRIBUTION-####################
#################################################################
## Number of projects created per programmer
project_creator_dist = analysis_df.groupBy('creator').count().toDF('creator','project_counts')
bins = [1,2,5,10,20,50,100]
project_creator_dist_rdd = project_creator_dist.map(lambda row: row.project_counts)
created_project_num = project_creator_dist_rdd.collect()
plt.gcf().clear()
plt.hist(created_project_num, bins, facecolor='b', alpha=0.75)
plt.xticks(np.arange(0, 101, 10.0))
plt.xlabel('Projects Created')
plt.ylabel('Frequency')
plt.title('Projects created per programmer')
plt.savefig(output_dir+'projects_created_histogram')

## population distribution
level_dist_pdf = norm_analysis_df.groupBy('total').count().toDF('Mastery Level', 'count').toPandas()
level_dist_pdf = level_dist_pdf.set_index('Mastery Level')
plt.gcf().clear()
level_dist_pdf.plot(kind='bar', color="b")
plt.xlabel('Mastery Level')
plt.ylabel('Frequency')
plt.title('Programming Mastery Score Distribution')
plt.savefig(output_dir+'mastery-distribution')

##size distribution
bloc_rdd = analysis_df.select('PS.bloc').rdd.map(lambda row: row.bloc)
size_max = 5000
bins = np.linspace(0, size_max, 15)
plt.gcf().clear()
plt.xlabel('BLOC')
plt.ylabel('frequency')
plt.title('Project Size (BLOC)')
plt.hist(bloc_rdd.filter(lambda x: x<=size_max).collect(), bins, facecolor='red', alpha=0.5)
plt.gca().set_xscale("linear")
plt.gca().set_yscale("log")
plt.savefig(output_dir+'bloc-dist')

##popularity distribution
favorites_rdd = analysis_df.select('favoriteCount').rdd.map(lambda row: row.favoriteCount)
favorites = favorites_rdd.collect()
loveCount = analysis_df.select('loveCount').rdd.map(lambda row: row.loveCount).collect()
views_rdd = analysis_df.select('views').rdd.map(lambda row: row.views)
views = views_rdd.collect()
bins = np.linspace(0, 1001, 100)

plt.gcf().clear()
plt.hist(views, bins=bins, facecolor='g', alpha=0.5, label='views')
plt.hist(loveCount, bins=bins, facecolor='b', alpha=0.5, label='loveCount')
plt.hist(favorites, bins=bins, facecolor='r', alpha=0.5, label='favoriteCount')
plt.xticks(np.arange(0, 101, 10.0))
plt.gca().set_xscale("linear")
plt.gca().set_yscale("log")
plt.legend(loc='upper right')
plt.ylabel('Frequency')
plt.title('Popularity of programs')
plt.savefig(output_dir+'popularity_histogram')

bins = [0,1,5,10,100]
zero_favorite_count = favorites_rdd.histogram(bins)[1][0]  
one_to_five_view_count = views_rdd.histogram(bins)[1][1]
print 'zero favrotite count:', zero_favorite_count, 'percent:', zero_favorite_count/original_count*100 #552812 , 82.23%
print '1-5 view count:', one_to_five_view_count, 'percent:', one_to_five_view_count/original_count*100 #556684 , 82.81%
###############################################################################################
#########################-STATISTICS-OF-SMELLS-###############################################
##############################################################################################
smell_stats_pdf = norm_analysis_df.select(smells).describe().toPandas().transpose()
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

#############################Distinct Smells######################################
#Summary Smell Stats
combined_stats_pdf = average_smells_per_script_pdf.join(percentage_smell_pdf).round(2)
combined_stats_pdf.columns.name='Smell'
write_latex(combined_stats_pdf, output_dir+'smell_stats.tex')
#Comparison
current_palette = sns.color_palette("husl", 10)
metrics = ['PS.script', 'PS.sprite','PS.bloc', 'PS.script_length']
metrics_pdf = analysis_df.groupBy('MS.total').avg(*metrics).toDF('Mastery Level', *metrics).toPandas()
metrics_pdf = metrics_pdf.set_index('Mastery Level')
metrics_pdf = metrics_pdf.plot(kind='bar', color=current_palette)
plt.savefig(output_dir+'mastery-metrics')

###############################################################################################

smells_pdf = norm_analysis_df.groupBy('MS.total').avg(*smells).toDF('Mastery Level',*smells).toPandas()
smells_pdf = smells_pdf.set_index('Mastery Level')
smells_pdf.plot(kind='bar', stacked=True, color=current_palette)
plt.savefig(output_dir+'mastery-smell')

##################Experiment###################
ii_pdf = norm_analysis_df.select('II', 'level').groupBy('level').avg('II').toDF('Mastery Level', 'II').toPandas()
ii_pdf = ii_pdf.set_index('Mastery Level')
plt.gcf().clear()
ii_pdf.plot(kind='bar', color=current_palette)
plt.xlabel('Mastery Level')
plt.ylabel('Smell Density')
plt.savefig(output_dir+'II')

#aggregate by two factors

ii_pdf = norm_analysis_df.select('II', 'level', udf_bloc_level(norm_analysis_df['PS.bloc']).alias('bloc')).groupBy('bloc', 'level').agg({"II":"avg"}).toDF( 'bloc','level', 'avgII').toPandas()
plt.gcf().clear()
ii_pdf.plot(kind='bar', color=current_palette)
plt.xlabel('Mastery Level')
plt.ylabel('Smell Density')
plt.savefig(output_dir+'IIByBloc')

ii_pdf_t = ii_pdf.transpose()
ii_pdf_t.columns = [ii_pdf_t.iloc[0],ii_pdf_t.iloc[1]]
ii_pdf_t = ii_pdf_t.reindex(ii_pdf_t.index.drop('bloc'))
ii_pdf_t = ii_pdf_t.reindex(ii_pdf_t.index.drop('level'))

plt.gcf().clear()
ii_pdf_t.plot(kind='bar', color=current_palette)
plt.xlabel('Mastery Level')
plt.ylabel('Smell Density')
plt.savefig(output_dir+'IIByBloc')


##################################
##Threshold Table Calculation#####
##################################
def weight_pdf_to_vals(val_weight_pdf, val_colname, weight_colname):
  weighted = []
  for i, row in val_weight_pdf.iterrows():
    for j in range(int(row[weight_colname]*1000000)):
      weighted.append(row[val_colname])
  return weighted

total_bloc = analysis_df.select('PS.bloc').groupBy().sum().toDF('sum').take(1)[0].sum
analysis_df = analysis_df.withColumn('per_total_bloc', analysis_df['PS.bloc']/total_bloc)

table = []
threshold_headers=["Smell", "70%", "80%", "90%"]

smell_percent_pdf_dict = {}
for smell in smells:
   smell_percent_df = analysis_df.groupBy(smell+'.count').sum('per_total_bloc').toDF(smell, 'percent')
   smell_percent_pdf = smell_percent_df.toPandas()
   smell_percent_pdf_dict[smell] = smell_percent_pdf

for smell in smells:
  smell_vals = weight_pdf_to_vals(smell_percent_pdf_dict[smell], val_colname=smell, weight_colname='percent')
  row = []
  row.append(smell)
  for percent in [70,80,90]:
    row.append(np.percentile(smell_vals, percent, interpolation='higher'))
  table.append(row)

print tabulate(table, headers=threshold_headers)
latex_threshold_table = tabulate(table, headers, tablefmt="latex_booktabs")
write_to_file(latex_threshold_table ,output_dir+'threshold_table.tex')

#######################
##Percentage risk #####
#######################
risk_table = []
risk_headers = ["Smell", "> 70%", "> 80%", "> 90%"]
total_projects = analysis_df.count()

for threshold_row in table:
  smell = threshold_row[0]
  threshold_range = threshold_row[1:4]
  project_percent_row = []
  project_percent_row.append(smell)
  for threshold_val in threshold_range:
    percent = (analysis_df.filter(analysis_df[smell+'.count']>threshold_val).count()/total_projects)*100
    project_percent_row.append("{0:.2f}%".format(percent))
  risk_table.append(project_percent_row)

print tabulate(risk_table, headers=risk_headers)
latex_smell_percentage_table = tabulate(risk_table, headers=risk_headers, tablefmt="latex_booktabs")
write_to_file(latex_smell_percentage_table ,output_dir+'smell_percentage_table.tex')
########################

for percent in [70,80,90]:
  threshold_bins.append(np.percentile(w, percent, interpolation='higher'))



total_projects = weighted_sample_df.count()
project_percent_row = []
for threshold in threshold_bins:
  project_percent_row.append(weighted_sample_df.filter(weighted_sample_df['val']>threshold).count()/total_projects)

########################################################################################
############# SMELL ANALYSIS FOR EACH CATEGORY #########################################
########################################################################################

# each_smell_density_pdf = norm_analysis_df.select(smells+['masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg(*smells).toDF('Project Size', 'Mastery level', *smells).toPandas()
################################
#---Dead Code (US, UV, UCB)----#
################################
dead_code_smells = ['US','UV', 'UCB']
dead_code_smells_dict = {'US':'Unreachable Script', 'UV':'Unused Variable', 'UCB':'Unused Custom Block'}
dead_code_smells_ylabels = {'US': 'smell/BLOC', 'UV':'smell/total vars', 'UCB':'smell/total custom blocks'}
dead_code_df = analysis_df.select([col for col in analysis_df.columns if col not in diff(smells,dead_code_smells)])
#% of total_bloc
dead_code_df = dead_code_df.withColumn('per_total_bloc', dead_code_df['PS.bloc']/total_bloc)


##normalize readability code smells
dead_code_df = dead_code_df.withColumn('US',udf_normalize(dead_code_df['US.count'], dead_code_df['PS.bloc']))
dead_code_df = dead_code_df.withColumn('UV',udf_normalize(dead_code_df['UV.count'], dead_code_df['PE.varCount']))
dead_code_df = dead_code_df.withColumn('UCB',udf_normalize(dead_code_df['UCB.count'], dead_code_df['PE.customBlockCount']))
##aggregation
dead_code_analysis_pdf = dead_code_df.select(dead_code_smells+['masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg(*dead_code_smells).toDF('Project Size', 'Mastery level', *dead_code_smells).toPandas()
##Dead Code vs Size vs Mastery##
plt.gcf().clear()
colNum = 3; rowNum = 1
fig, axes = plt.subplots(nrows=rowNum, ncols=colNum, figsize=(12, 4))
col = 0
for smell in dead_code_smells:
   ax = sns.barplot(x="Project Size", y=smell, hue="Mastery level", data=dead_code_analysis_pdf, palette="Blues", ax=axes[col])
   ax.set_title(dead_code_smells_dict[smell])
   ax.set_ylabel(dead_code_smells_ylabels[smell])
   col = col+1

plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.0)
plt.savefig(output_dir+'DeadCode-size-mastery-comparison')

##density_threshold
dead_code_pdf = dead_code_df.select(dead_code_smells).toPandas()
# test_dead_code_pdf[np.isfinite(test_dead_code_pdf['US'])]
dead_code_thresholds = dead_code_pdf.quantile([.7,.8,.9])

#determine threshold

us_percent_per_bloc = analysis_df.select(udf_normalize(analysis_df['US.count'], analysis_df['PS.bloc']))

# test flatten weighted sample to dataframe
weighted_sample_df = sqlContext.createDataFrame([{'val':0, 'loc':70},{'val': 1, 'loc': 10},{'val': 2, 'loc': 10}, {'val': 3, 'loc': 5}, {'val': 4, 'loc': 5}], ['val', 'loc'])
# toPercent = lambda smell, total: smell/total
# udf_norm_smell = udf(normalized_by_script, DoubleType())
total = weighted_sample_df.select('loc').groupBy().sum().toDF('sum').take(1)[0].sum
weighted_sample_df = weighted_sample_df.withColumn('per_total_bloc', weighted_sample_df['loc']/total)
val_bloc_percent_df = weighted_sample_df.groupBy('val').sum('per_total_bloc').toDF('val', 'percent')
val_bloc_percent_pdf = val_bloc_percent_df.toPandas()
w = weight_pdf_to_vals(val_bloc_percent_pdf, val_colname='val', weight_colname='percent')
threshold_bins = []


for percent in [70,80,90]:
  threshold_bins.append(np.percentile(w, percent, interpolation='higher'))



total_projects = weighted_sample_df.count()
project_percent_row = []
for threshold in threshold_bins:
  project_percent_row.append(weighted_sample_df.filter(weighted_sample_df['val']>threshold).count()/total_projects)
  

# percent_project = np.histogram(w,bins=threshold_bins)[0]


vals = sqlContext.createDataFrame([{'val': 1},
  {'val': 2,}, {'val': 4}])
weighted_sample_pdf = weighted_sample_df.toPandas()


def weight_array(val_weight_pdf, val_colname, weight_colname):
  weighted = []
  for i, row in val_weight_pdf.iterrows():
    for j in range(int(row[weight_colname]*100)):
      weighted.append(row[val_colname])
  return weighted

w = weight_array(weighted_sample_pdf, val_colname='val', weight_colname='percent')
for percent in [70,80,90]:
  print np.percentile(w, percent)




# test_dead_code_df = sqlContext.createDataFrame([{'_id': '123', 'bloc': 20, 'US':0, 'UV':4, 'UCB':1, 'bloc':40.0},
#   {'_id': '234', 'bloc': 10, 'US':1, 'UV':2, 'UCB':3, 'bloc':50.0}])

# df = pd.DataFrame(np.array([[1], [2], [3], [4]]), columns=['a'])

# adj_test_dead_code_df = test_dead_code_df.select(*[udf_normalize(column, test_dead_code_df['bloc']).alias(column) if column in ['US','UV'] else column for column in test_dead_code_df.columns])
# test_dead_code_pdf = test_dead_code_df.toPandas()
# test_dead_code_pdf[np.isfinite(test_dead_code_pdf['US'])]


###EXAMPLE Adjustment###
# var_smell_analysis_df = norm_analysis_df.select(*[udf_normalize(column, norm_analysis_df['PE.varCount']).alias(column) if column in var_related_smells else column for column in norm_analysis_df.columns])

#example adjust by bloc
example_bloc_adj_df = var_smell_analysis_df.select(*[udf_normalize(column+'.count', var_smell_analysis_df['PS.bloc']).alias(column) if column in var_related_smells else column for column in var_smell_analysis_df.columns])
example_bloc_adj_df = example_bloc_adj_df.select(['UV','masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg('UV').toDF('Project Size', 'Mastery level', 'UV').toPandas()
plt.gcf().clear()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
#unadj
ax0 = sns.barplot(x="Project Size", y='UV', hue="Mastery level", data=example_bloc_adj_df, palette="Blues", ax=axes[0])
ax0.set_title('Unused Variable')
ax0.set_ylabel('Smells/BLOC')
ax0.get_legend().loc="upper right"
#adj
ax1 = sns.barplot(x="Project Size", y='UV', hue="Mastery level", data=dead_code_analysis_pdf, palette="Blues", ax=axes[1])
ax1.set_title('Unused Variable')
ax1.set_ylabel('Smells/total_var (Adjusted)')

plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.0)
plt.savefig(output_dir+'adjst_vs_unadjusted')

# Example Relationship Between Number of Variables and Project Size
var_proj_relationship_pdf = norm_analysis_df.select('PS.bloc', 'PE.varCount').toDF('BLOC', 'Var').toPandas()
var_proj_relationship_pdf.plot(kind='scatter', x='BLOC',y='Var', s=0.1)
var_proj_relationship_pdf[['BLOC','Var']].corr()
plt.savefig(output_dir+'BLOC-Var-relation')
print 'correlation between BLOC and Var:', var_proj_relationship_pdf[['BLOC','Var']].corr()['BLOC']['Var'] #0.49
# Example Inverse Relationship Between Unused Variable and Total Variable
###END EXAMPLE###

############################################
#-----Duplication ['DC', 'DV', 'HCMS']-----#
############################################
##normalize duplicate code###
duplication_analysis_df = analysis_df.select([col for col in analysis_df.columns if col not in diff(smells,['DC','DV','HCMS'])])
duplication_analysis_df = duplication_analysis_df.filter(duplication_analysis_df['DC.count']>0)
duplication_analysis_df = duplication_analysis_df.filter(duplication_analysis_df['DV.stringCount']>0)
duplication_analysis_df = duplication_analysis_df.withColumn('DC(same-sprite)',udf_normalize(duplication_analysis_df['DC.sameSpriteClone'], duplication_analysis_df['PS.bloc']))
duplication_analysis_df = duplication_analysis_df.withColumn('DC(inter-sprite)',udf_normalize(duplication_analysis_df['DC.interSpriteClone'], duplication_analysis_df['PS.bloc']))
duplication_analysis_df = duplication_analysis_df.withColumn('DV(string)',udf_normalize(duplication_analysis_df['DV.stringCount'], duplication_analysis_df['PS.bloc']))
duplication_analysis_df = duplication_analysis_df.withColumn('HCMS',udf_normalize(duplication_analysis_df['HCMS.count'], duplication_analysis_df['PS.bloc']))
#############################

duplication_smells = ['DC(same-sprite)','DC(inter-sprite)', 'DV(string)', 'HCMS']
compare_duplication =  ['DC(same-sprite)', 'DV(string)', 'HCMS']
duplication_smells_dict = {'DC(same-sprite)':'Duplicate Code (same-sprite)','DC(inter-sprite)':'Duplicate Code (inter-sprite)', 'DV(string)':'Duplicate Value (string)', 'HCMS':'Hard-Coded Media Sequence'}
duplication_analysis_pdf = duplication_analysis_df.select(duplication_smells+['masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg(*duplication_smells).toDF('Project Size', 'Mastery level', *duplication_smells).toPandas()

plt.gcf().clear()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
col = 0
for smell in compare_duplication:
   ax = sns.barplot(x="Project Size", y=smell, hue="Mastery level", data=duplication_analysis_pdf, palette="Blues", ax=axes[col])
   ax.set_title(duplication_smells_dict[smell])
   ax.set_ylabel('smells/BLOC')
   col = col+1

plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.0)
plt.savefig(output_dir+'duplication-size-mastery-comparison')

#within vs inter-sprite
DC_same_vs_inter_comparison = ['DC(same-sprite)', 'DC(inter-sprite)']
plt.gcf().clear()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
col = 0
for smell in DC_same_vs_inter_comparison:
   ax0 = sns.barplot(x="Project Size", y=duplication_smells[col], hue="Mastery level", data=duplication_analysis_pdf, palette="Blues", ax=axes[col])
   ax0.set_title(duplication_smells[col])
   ax0.set_ylabel('Smells/BLOC')
   ax0.set_ylim([0,0.1])
   col = col+1

plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.0)
plt.savefig(output_dir+'same-sprite-vs-inter-sprite')

#? How many percent is duplicate substring

############################################
#-------Readability [TL, UN, TFGS] --------#
############################################
readability_smells = ['TLS', 'UN', 'TFGS']
readability_smells_dict = {'TLS':'Too Long Script', 'UN':'Uncommunicative Naming', 'TFGS':'Too-Fine-Grained Script'}
readability_smells_ylabels = {'TLS':'smells / bloc', 'UN':'smells / scriptable', 'TFGS':'smells / bloc'}
readability_analysis_df = analysis_df.select([col for col in analysis_df.columns if col not in diff(smells,readability_smells)])
##normalize readability code smells
readability_analysis_df = readability_analysis_df.withColumn('TLS',udf_normalize(readability_analysis_df['TLS.count'], duplication_analysis_df['PS.bloc']))
readability_analysis_df = readability_analysis_df.withColumn('UN',udf_normalize(readability_analysis_df['UN.count'], duplication_analysis_df['PS.sprite']))
readability_analysis_df = readability_analysis_df.withColumn('TFGS',udf_normalize(readability_analysis_df['TFGS.count'], duplication_analysis_df['PS.bloc']))
##aggregation
readability_analysis_pdf = readability_analysis_df.select(readability_smells+['masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg(*readability_smells).toDF('Project Size', 'Mastery level', *readability_smells).toPandas()
##plot
plt.gcf().clear()
colNum = 3; rowNum = 1
fig, axes = plt.subplots(nrows=rowNum, ncols=colNum, figsize=(12, 4))
col = 0
for smell in readability_smells:
   ax = sns.barplot(x="Project Size", y=smell, hue="Mastery level", data=readability_analysis_pdf, palette="Blues", ax=axes[col])
   ax.set_title(readability_smells_dict[smell])
   ax.set_ylabel(readability_smells_ylabels[smell])
   col = col+1

plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.0)
plt.savefig(output_dir+'Readability-size-mastery-comparison')

######################################################
#-------(Unnecessary Complexity [BCW, UB, BVS]-------#
######################################################
# BCW, UB, BVS
complexity_smells = ['BCW', 'UBC', 'BVS']
complexity_smells_dict = {'BCW':'Broadcast Workaround', 'UBC':'Unnecessary Broadcast', 'BVS': 'Broad Variable Scope'}
complexity_smells_ylabels = {'BCW':'smells / bloc', 'UBC':'smells / bloc', 'BVS':'smells / total variables'}
# column reduction
complexity_analysis_df = analysis_df.select([col for col in analysis_df.columns if col not in diff(smells,complexity_smells)])
#normalize complexity code smells
complexity_analysis_df = complexity_analysis_df.withColumn('BCW',udf_normalize(complexity_analysis_df['BCW.count'], complexity_analysis_df['PS.bloc']))
complexity_analysis_df = complexity_analysis_df.withColumn('UBC',udf_normalize(complexity_analysis_df['UBC.count'], complexity_analysis_df['PS.bloc']))
complexity_analysis_df = complexity_analysis_df.withColumn('BVS',udf_normalize(complexity_analysis_df['BVS.count'], complexity_analysis_df['PE.varCount']))
##aggregation
complexity_analysis_pdf = complexity_analysis_df.select(complexity_smells+['masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg(*complexity_smells).toDF('Project Size', 'Mastery level', *complexity_smells).toPandas()
##plot
plt.gcf().clear()
colNum = 3; rowNum = 1
fig, axes = plt.subplots(nrows=rowNum, ncols=colNum, figsize=(12, 4))
col = 0
for smell in complexity_smells:
   ax = sns.barplot(x="Project Size", y=smell, hue="Mastery level", data=complexity_analysis_pdf, palette="Blues", ax=axes[col])
   ax.set_title(complexity_smells_dict[smell])
   ax.set_ylabel(complexity_smells_ylabels[smell])
   col = col+1

plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.0)
plt.savefig(output_dir+'Complexity-size-mastery-comparison')


##################Effects of Code Smells##############
# smell_density_pdf = norm_analysis_df.select(['total smell','level','bloc level']).groupBy('bloc level', 'level').avg('total smell').toDF('bloc level', 'level', 'total smell').toPandas()
# Relationship between smell density v.s. project size and programmer's mastery level
smell_density_pdf = norm_analysis_df.select(['total smell','masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg('total smell').toDF('Project Size', 'Mastery level', 'Avg. Smells').toPandas()
plt.gcf().clear()
sns.barplot(x="Project Size", y="Avg. Smells", hue="Mastery level", data=smell_density_pdf, palette="Greens")
plt.xlabel('Project Size')
plt.ylabel('Average Smell Density')
plt.savefig(output_dir+'Overall-Smell-Mastery-Size')
##############Smell vs Project Size vs Mastery Level#####################
each_smell_density_pdf = norm_analysis_df.select(smells+['masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg(*smells).toDF('Project Size', 'Mastery level', *smells).toPandas()
plt.gcf().clear()
colNum = 3; rowNum = 4

fig, axes = plt.subplots(nrows=rowNum, ncols=colNum, figsize=(12, 18))
i = 0; j = 0
for smell in smells:
   if j!=0 and j%(colNum)==0:
      i=i+1
   ax = sns.barplot(x="Project Size", y=smell, hue="Mastery level", data=each_smell_density_pdf, palette="Greens", ax=axes[i%(rowNum),j%(colNum)])
   ax.set_title(smell)
   j=j+1

plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.0)
plt.savefig(output_dir+'each-smell-size-mastery-comparison')

