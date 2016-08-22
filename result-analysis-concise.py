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
from pyspark.sql.functions import explode

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
      dbhost = 'mongodb://hs008:27017/'
      format_dir = 'file:///home/tpeera4/projects/spark-smell-aggregate/format/'
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
analysis_df = project_metadata_df.join(reports_df, '_id')
analysis_df = analysis_df.join(metrics_df, '_id')
# analysis_df = analysis_df.join(metrics_df, '_id')
analysis_df = analysis_df.join(creators_df, 'creator')

###############Expand Column################################
analysis_df = analysis_df.withColumn('DC_inter', analysis_df['DC.interSpriteClone'])
analysis_df = analysis_df.withColumn('DC.count', analysis_df['DC.sameSpriteClone'])


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
# original_analysis_df = original_analysis_df.withColumn('totalSmell', sum(original_analysis_df[smell+'.count'] for smell in smells))

##############normalization#####################
##smell per bloc (block line of code)
densitySmells = ['DV','TFGS','TLS','DC']
expandedSmells = ['DC_inter']
absoluteSmells = diff(smells, densitySmells+expandedSmells)
def normalize(count, bloc):
  if count == None:
    return 0.0
  elif count==0:
    return 0.0
  else:
    return count/bloc

# normalize = lambda count, bloc: 0.0 if count == 0 else count/bloc

udf_normalize = udf(normalize, DoubleType())

get_count = lambda count: count
udf_get_count = udf(get_count, IntegerType())

norm_analysis_df = analysis_df.select(*[udf_normalize(column+'.count', analysis_df['PS.bloc']).alias(column) if column in densitySmells else column for column in analysis_df.columns])
# norm_analysis_df = norm_analysis_df.select(*[udf_normalize(column, norm_analysis_df['PS.bloc']).alias(column) if column in expandedSmells else column for column in norm_analysis_df.columns])
norm_analysis_df = norm_analysis_df.withColumn('DC_inter', udf_normalize(norm_analysis_df['DC_inter'], norm_analysis_df['PS.bloc']))
norm_analysis_df = norm_analysis_df.select(*[udf_get_count(column+'.count').alias(column) if column in absoluteSmells else column for column in norm_analysis_df.columns])
proficient_df = norm_analysis_df.filter(norm_analysis_df['masteryLevel']==3) 

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

###############################################################################################
#########################-STATISTICS-OF-SMELLS-###############################################
##############################################################################################
################Percent of each smells in the entire population##################
exists = lambda col: 1 if col > 0 else 0
udf_exists = udf(exists, IntegerType())
distinct_smell_df = norm_analysis_df.select(*[udf_exists(column).alias(column) for column in smells])
with_total_distinct_smells_df = distinct_smell_df.withColumn('Distinct Smell Counts', sum([distinct_smell_df[smell] for smell in smells]))
row_counts = with_total_distinct_smells_df.count()
found_smell_sum = with_total_distinct_smells_df.groupby().sum(*smells).toDF(*smells)
found_smell_sum_pdf = found_smell_sum.toPandas()
percentage_smell_pdf = found_smell_sum_pdf.applymap(lambda found: found/row_counts*100)
percentage_smell_pdf = percentage_smell_pdf.transpose()
percentage_smell_pdf.columns=['freq (%)']
write_latex(percentage_smell_pdf.round(2) ,output_dir+'percent_smell_found.tex')
###############################################################################################

##################################
##Threshold Table Calculation#####
##################################
def weight_pdf_to_vals(val_weight_pdf, val_colname, weight_colname):
  weighted = []
  for i, row in val_weight_pdf.iterrows():
    for j in range(int(row[weight_colname]*1000000)):
      weighted.append(row[val_colname])
  return weighted


num_proficient_level = proficient_df.count() # 232426 out of 880809

total_bloc = proficient_df.select('PS.bloc').groupBy().sum().toDF('sum').take(1)[0].sum
proficient_df = proficient_df.withColumn('per_total_bloc', proficient_df['PS.bloc']/total_bloc)

table = []
threshold_headers=["Smell", "70%", "80%", "90%"]

smell_percent_pdf_dict = {}
for smell in smells:
   clean_proficient_df = proficient_df.select([smell, 'per_total_bloc']).dropna()
   smell_percent_df = clean_proficient_df.groupBy(smell).sum('per_total_bloc').toDF(smell, 'percent')
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
latex_threshold_table = tabulate(table, threshold_headers, tablefmt="latex_booktabs")
write_to_file(latex_threshold_table ,output_dir+'threshold_table.tex')

##############################
######Percentage risk ########
##############################
risk_table = []
risk_headers = ["Smell", "> 70%", "> 80%", "> 90%"]


for threshold_row in table:
  smell = threshold_row[0]
  clean_proficient_df = norm_analysis_df.select(smell).dropna()
  total_projects = clean_proficient_df.count()
  threshold_range = threshold_row[1:4]
  project_percent_row = []
  project_percent_row.append(smell)
  for threshold_val in threshold_range:
    percent = (clean_proficient_df.filter(clean_proficient_df[smell]>threshold_val).count()/total_projects)*100
    project_percent_row.append("{0:.2f}%".format(percent))
  risk_table.append(project_percent_row)

print tabulate(risk_table, headers=risk_headers)
latex_smell_percentage_table = tabulate(risk_table, headers=risk_headers, tablefmt="latex_booktabs")
write_to_file(latex_smell_percentage_table ,output_dir+'smell_percentage_table.tex')

########################################################################################
############# SMELL ANALYSIS FOR EACH CATEGORY #########################################
########################################################################################

################################
#---Dead Code (US, UV, UCB)----#
################################
dead_code_smells = ['US','UV', 'UCB']
dead_code_smells_dict = {'US':'Unreachable Script', 'UV':'Unused Variable', 'UCB':'Unused Custom Block'}
dead_code_smells_ylabels = {'US': 'smell', 'UV':'smell', 'UCB':'smell'}
dead_code_df = norm_analysis_df.select([col for col in norm_analysis_df.columns if col not in diff(smells,dead_code_smells)])
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


############################################
#-----Duplication ['DC', 'DV', 'HCMS']-----#
############################################
duplication_smells = ['DC','DV','HCMS']
#############################
duplication_smells_dict = {'DC':'Duplicate Code (same-sprite)','DC_inter':'Duplicate Code (inter-sprite)','DV':'Duplicate Value (string)', 'HCMS':'Hard-Coded Media Sequence'}
duplication_analysis_pdf = norm_analysis_df.select(duplication_smells+['masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg(*duplication_smells).toDF('Project Size', 'Mastery level', *duplication_smells).toPandas()
duplication_smells_ylabels = {'DC':'smell/bloc', 'DV':'smell/bloc', 'HCMS':'smell'}
plt.gcf().clear()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
col = 0
for smell in duplication_smells:
   ax = sns.barplot(x="Project Size", y=smell, hue="Mastery level", data=duplication_analysis_pdf, palette="Blues", ax=axes[col])
   ax.set_title(duplication_smells_dict[smell])
   ax.set_ylabel(duplication_smells_ylabels[smell])
   col = col+1

plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.0)
plt.savefig(output_dir+'duplication-size-mastery-comparison')

#within vs inter-sprite
DC_same_vs_inter_comparison = ['DC', 'DC_inter']
dup_same_vs_inter_analysis_pdf = norm_analysis_df.select(DC_same_vs_inter_comparison+['masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg(*DC_same_vs_inter_comparison).toDF('Project Size', 'Mastery level', *DC_same_vs_inter_comparison).toPandas()

plt.gcf().clear()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
col = 0
for smell in DC_same_vs_inter_comparison:
   ax0 = sns.barplot(x="Project Size", y=smell, hue="Mastery level", data=dup_same_vs_inter_analysis_pdf, palette="Blues", ax=axes[col])
   ax0.set_title(DC_same_vs_inter_comparison[col])
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
readability_smells_ylabels = {'TLS':'smells / bloc', 'UN':'smell', 'TFGS':'smells'}
##aggregation
readability_analysis_pdf = norm_analysis_df.select(readability_smells+['masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg(*readability_smells).toDF('Project Size', 'Mastery level', *readability_smells).toPandas()
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
complexity_smells_ylabels = {'BCW':'smell', 'UBC':'smell', 'BVS':'smell'}
# column reduction
complexity_analysis_df = norm_analysis_df.select([col for col in norm_analysis_df.columns if col not in diff(smells,complexity_smells)])
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

############### Threshold for TLS ############

bloc_df = proficient_df.select(explode(proficient_df['PS.blocPerScript']))
bloc_rdd = bloc_df.rdd.map(lambda row: row.col)
bloc_array = bloc_rdd.collect()
np.percentile(bloc_array,[70,80,90]) # 5,7,11

############### Threshold for Purity #############
purity_df = proficient_df.select(explode(proficient_df['SO.purity_vals']))
purity_rdd = purity_df.rdd.map(lambda row: row.col)
purity_array = purity_rdd.collect()
np.percentile(purity_array,[10,20,30])  #0.5 ~ 1/3 (20th percentile) at least 80% of the values has higher purity value than 1/3

percent = (norm_analysis_df.filter(norm_analysis_df['SO.avg']< 0.5).count()/total_projects)*100 #38.97% of projects have lower purity value than 1/3
script_organization_pdf = norm_analysis_df.select(['SO','masteryLevel','bloc level']).groupBy('bloc level','masteryLevel').avg('SO.avg').toDF('Project Size', 'Mastery level', "Purity").toPandas()
sns.barplot(x="Project Size", y=smell, hue="Mastery level", data=script_organization_pdf, palette="Blues", ax=axes[col])