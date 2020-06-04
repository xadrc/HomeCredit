"""
    LIBRAIRIES
"""

import findspark ;      findspark.init()
import pyspark

import os
import itertools
import argparse

import pandas            as pd
import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt

from collections                import namedtuple
from tabulate                   import tabulate

from pyspark.sql                import SparkSession
from pyspark.sql.functions      import when, isnull, mean, round, col
from pyspark.ml                 import Pipeline
from pyspark.ml.feature         import OneHotEncoder,       \
                                       StringIndexer,       \
                                       VectorAssembler,     \
                                       Imputer
from pyspark.ml.classification  import LogisticRegression
from pyspark.ml.tuning          import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation      import BinaryClassificationEvaluator,       \
                                       MulticlassClassificationEvaluator

from sklearn.metrics            import classification_report, confusion_matrix


"""
    PROCESSING DATA
"""

def csv_to_sparkdf(data_path, verbose = True):
    """
    Import data.csv --> Spark dataframe
    """
    sparkdf = spark.read.csv(
        data_path,
        header      = True,
        inferSchema = True
    )
    if verbose: 
        print('\nLOADED DATA SET:\n') ; sparkdf.printSchema()

    return sparkdf


"""
    DATA WRANGLING
"""

def drop_col(sparkdf, list_dropcol):
    """
    Drop unused columns
        - sparkdf:      Spark dataframe
        - list_dropcol: List of column names to be dropped 
    """
    return sparkdf.select(
        [c for c in sparkdf.columns if c not in list_dropcol]
    )


def calc_classratio(sparkdf, target_col, verbose = True, plot = True):
    """
    Calculate ratio `cat_1` / `cat_2`
        - sparkdf:     Spark dataframe
        - target_col:  name of the target column
        - verbose:     print result
        - plot:        visualise balance on histogramm
    """
    dist = sparkdf\
        .groupby(target_col)\
        .count()\
        .toPandas()

    ratio = (1 - (dist.loc[0, ['count']][0] / dist.loc[1, ['count']][0]))
    
    if verbose:
        print('IMBALANCED CLASS RATIO:\n')
        print('{}\n'.format(ratio.round(2)))
    
    if plot:
        d = sparkdf.select('TARGET').toPandas()
        plt.figure(figsize = (10, 10))
        sns.countplot(
            x      = target_col, 
            order  = d[target_col].value_counts().index,
            data   = d, 
            orient = 'h'
        )
        plt.savefig(
            fname  = os.path.join(dir_outp, 'class_balance.pdf'),
            dpi    = 300,
            format = 'pdf'
        )
        plt.close()

    return ratio

 
def count_missing(sparkdf, verbose = True):
    """
    Count number of missing values per features in dataframe
        - sparkdf:  Spark dataframe
    """
    features = []
    # count n missing values for each feature in sparkdf
    for col in sparkdf.dtypes:    
        col_name  = col[0]
        n_missval = sparkdf.where(sparkdf[col_name].isNull()).count()
        features.append(tuple([col_name, n_missval]))
    # filter col only countaining missing values
    list_misscol = [(x, y) for (x, y) in features if y != 0]

    if verbose:
        print('COUNT MISSING VALUES PER FEATURES:\n')
        print(*list_misscol, sep = '\n')
        print('\n')

    return list_misscol


def info_missing(sparkdf, verbose = True):
    """
    Insights about missing values in dataframe
        - sparkdf:  Spark dataframe
    """
    pandasdf = sparkdf.toPandas()
    # total count missing values / feat
    mis_val = pandasdf.isnull().sum()
    if mis_val is None:
        print("Data set contains no missing values")
        return None
    else:
        mis_val_percent = 100 * pandasdf.isnull().sum() / len(pandasdf)
        # create table
        mis_val_table  = pd.concat(
            [mis_val, mis_val_percent], 
            axis = 1
        )
        # rename col
        mis_val_table = mis_val_table.rename(
            columns = {
                0 : 'n_missval',
                1 : 'perc_total'
            }
        )
        mis_val_table = mis_val_table\
            [mis_val_table.iloc[:,1] != 0]\
            .sort_values('perc_total', ascending = False)\
            .round(2)     

        if verbose: 
            print('SUMMARY MISSING VARIABLES:\n')
            table = tabulate(
                mis_val_table,
                headers  = ['variable name', 'n missing', '%'],
                tablefmt = 'psql'
            )
            print(table, '\n')

        return mis_val_table


def filter_cn(sparkdf, verbose = True):
    """
    Separates categorical features from numerical features in dataframe
    Returns a named tuple `("features", ["categorical", "numerical"])`
        - sparkdf:  Spark dataframe
    """
    ft_cat  = [ft[0] for ft in sparkdf.dtypes if ft[1].startswith('string')] 
    ft_num  = [ft[0] for ft in sparkdf.dtypes if ft[1].startswith('int') | ft[1].startswith('double')][1:]

    ft = namedtuple("features", ["categorical", "numerical"])
    ft = ft(ft_cat, ft_num)

    if verbose:
        table = tabulate(
            [
                ['Categorical', len(ft.categorical)], 
                ['Numerical',   len(ft.numerical)  ]
            ],
            headers  = ['Features', 'count'],
            tablefmt = 'psql'
        )
        print('VARIABLE TYPES:\n') ; print(table, '\n')

    return ft


def fill_cat_ft(sparkdf, list_cat_ft, list_miss_ft, verbose = True):
    """
    Fills categorical features in spark dataframe with most frequent category
        - sparkdf:      Spark dataframe
        - list_cat_ft:  list of categorical variable names
        - list_miss_ft: list of features containing missing values
    """
    table          = []
    miss_cat_ft    = [ft for ft in list_cat_ft if ft in list_miss_ft]
    sparkdf_nomiss = sparkdf.na.drop()

    for ft in miss_cat_ft:
        most_feq = sparkdf_nomiss.groupBy(ft).count().sort(col("count").desc()).collect()[0][0] 
        if most_feq is not None:
            sparkdf = sparkdf.na.fill({ft : most_feq})
            table.append([ft, most_feq])
    
    if verbose:
        print("FILLING MISSING CATEGORICAL VARIABLES:\n")
        table = tabulate(
            table,
            headers  = ["Name", "Replaced by (most frequent)"],
            tablefmt = 'psql'
        )
        print(table, '\n')
    
    return sparkdf


def fill_num_ft(sparkdf, list_num_ft, list_miss_ft, verbose = True):
    """
    Fills numerical features in spark dataframe with average value
        - sparkdf:      Spark dataframe
        - list_cat_ft:  list of numerical variable names
        - list_miss_ft: list of features containing missing values
    """
    table          = []
    miss_num_ft    = [ft for ft in list_num_ft if ft in list_miss_ft]
    sparkdf_nomiss = sparkdf.na.drop()

    for ft in miss_num_ft:
        mean_val = sparkdf_nomiss.select(round(mean(ft))).collect()[0][0] 
        sparkdf  = sparkdf.na.fill({ft : mean_val})
        table.append([ft, mean_val])

    if verbose:
        print("FILLING MISSING NUMERICAL VARIABLES:\n")
        table = tabulate(
            table,
            headers  = ["Name", "Replaced by (AVG)"],
            tablefmt = 'psql'
        )
        print(table, '\n')
    
    return sparkdf


def add_col_weight(sparkdf, ratio, verbose = True):
    """
    Adds a new column WEIGHT in spark dataframe based on class balance ratio
    """
    sparkdf = sparkdf.withColumn(
        'WEIGHT',
        when(sparkdf.TARGET == 1, ratio).otherwise(1 - ratio)
    )
    if verbose:
        print("ADDING WEIGHT COLUMN:\n")
        sparkdf.select('TARGET', 'WEIGHT').show(5)

    return sparkdf


def cat_to_onehotvector(sparkdf, list_cat_ft, verbose = True):
    """
    Converts categorical features in spark dataframe into OneHot Vectors
        - sparkdf:      Spark dataframe
        - list_cat_ft:  list of categorical variable names
    """
    stages   = []
    features = sparkdf.columns
    list_num_ft = [ft for ft in features if ft not in list_cat_ft and ft != 'TARGET']
    
    for cat_ft in list_cat_ft:
        string_idx = StringIndexer(
            inputCol  = cat_ft,
            outputCol = cat_ft + 'Index'
        )
        encoder = OneHotEncoder(
            inputCols  = [string_idx.getOutputCol()],
            outputCols = [cat_ft + "class_vec"]
        )
        stages += [string_idx, encoder]
    
    assembler = VectorAssembler(
        inputCols = [cat_ft + "class_vec" for cat_ft in list_cat_ft] + list_num_ft,
        outputCol = "FEATURES"
    )
    stages += [assembler]
    
    # apply stages of tranformation to spark dataframe
    pipeline      = Pipeline(stages = stages)
    pipelinemodel = pipeline.fit(sparkdf)
    sparkdf       = pipelinemodel.transform(sparkdf)

    if verbose:
        print("CONVERTING CATEGORICAL FEATURES TO ONEHOT VECTORS:\n")
        sparkdf.select('FEATURES').show(5)

    return sparkdf
    
    
"""
    MACHINE LEARNING
"""

def split(sparkdf, weights):
    """
    Splits spark dataframe into train & test sets
        - sparkdf: spark dataframe
        - weights: vector of weights for split
    """
    x, y = sparkdf.randomSplit(
        weights, 
        seed = 42
    )
    return x, y


def LR_fit(sparkdf_train, features, label, weight, verbose = True): 
    """
    Trains a logistic regression model to data
        - sparkdf_train:    training spark dataframe set
        - features:         column name of vector features
        - label:            comumn name of target
        - weight:           column name of classes weights
    """
    model = LogisticRegression(
        featuresCol = features, 
        labelCol    = label, 
        weightCol   = weight,
        maxIter     = 10
    )
    model = model.fit(sparkdf_train)

    if verbose:
        print("FITTING MODEL: {}\n".format(model.uid.split('_')[0].upper()))

    return model


def model_eval(model, sparkdf_test, label, weight, verbose = True, plot = True):
    """
    Evaluates binary classification model
        - model             binary classification model
        - sparkdf_test:     testing spark dataframe set
        - label:            comumn name of target
        - weight:           column name of classes weights
    """
    predictions = model.transform(sparkdf_test)
    evaluator   = BinaryClassificationEvaluator(
        labelCol  = label,
        weightCol = weight
    )

    conf_mat = confusion_matrix(
        predictions.select(label).collect(),        # label real
        predictions.select('prediction').collect()  # label predicted
    )
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis = 1)[:, np.newaxis]

    name      = model.uid.split('_')[0].upper()
    precision = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])  # = TP/(TP+FP)
    recall    = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])  # = TP/(TP+FN)
    f1_score  = 2 * ((precision * recall) / (precision + recall))
    auc       = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

    if verbose:
        print("EVALUATING MODEL: {}\n".format(name))
        # print confusion matrix
        table_1 = tabulate(
            [ ["Positive", conf_mat[0][0], conf_mat[0][1] ],
              ["Negative", conf_mat[1][0], conf_mat[1][1] ] ],
            headers  = ["•", "Positive", "Negative"],
            tablefmt = 'psql'
        )        
        print("Confusion matrix:")
        print(table_1, '\n')
        # print model metrics
        table_2 = tabulate(
            [
                ["Precision", precision], 
                ["Recall", recall], 
                ["F1 score", f1_score], 
                ["Area under ROC", auc]
            ],
            headers = ['•', 'Value'],
            tablefmt = 'psql'
        )
        print("Metrics:")
        print(table_2, '\n')
    
    if plot:
        # Export ROC curve
        roc  = model.summary.roc.toPandas()
        plt.figure(figsize = (10, 10))
        plt.plot(roc['FPR'], roc['TPR'])
        plt.title('ROC (LOGISTIC REGRESSION)')
        plt.ylabel('False Positive Rate')
        plt.xlabel('True Positive Rate')
        plt.savefig(
            fname  = os.path.join(dir_outp, 'ROC_{}.pdf'.format(name)),
            dpi    = 300,
            format = 'pdf'
        )
        plt.close()
        # Export confusion matrix
        classes = ['Positive', 'Negative']
        ticks   = np.arange(model.numClasses)
        fmt     = '.2f'
        thresh  = conf_mat.max() /2.
        plt.figure(figsize = (10, 10))
        plt.imshow(conf_mat, interpolation = 'nearest', cmap = plt.cm.Blues)
        plt.colorbar()
        plt.xticks(ticks, classes, rotation = 45)
        plt.yticks(ticks, classes)
        for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            plt.text(
                j, i, 
                format(conf_mat[i, j], fmt),
                horizontalalignment = 'center',
                color = 'red' if conf_mat[i, j] > thresh else 'green'
            )
        plt.ylabel('Real label') ; plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(
            fname  = os.path.join(dir_outp, 'CM_{}.pdf'.format(name)),
            dpi    = 300, 
            format = 'pdf'
        )   
        plt.close()

    model_metrics = namedtuple("model", ["name", "confusion_matrix", "precision", "recall", "f1_score", "auc"])
    model_metrics = model_metrics(name, conf_mat, precision, recall, f1_score, auc)

    return model_metrics


"""
    MAIN()
"""

def main():
    df = csv_to_sparkdf(os.path.join(dir_data, 'application.csv'))
    df = drop_col(df, ['SK_ID_CURR', 'CODE_GENDER'])
    ratio = calc_classratio(df, 'TARGET')
    missing_info = info_missing(df)
    ft_types     = filter_cn(df)
    df = fill_cat_ft(df, ft_types.categorical, missing_info.index.tolist())
    df = fill_num_ft(df, ft_types.numerical, missing_info.index.tolist())
    df = add_col_weight(df, ratio)
    df = cat_to_onehotvector(df, ft_types.categorical)
    df_train, df_test = split(df, [.80, .20])
    LR_model   = LR_fit(df_train,  'FEATURES', 'TARGET', 'WEIGHT')
    LR_eval    = model_eval(LR_model, df_test, 'TARGET', 'WEIGHT')

    return None


"""
    EXECUTE 
"""

if __name__ == '__main__':
    # CONF SPARK SESSION
    findspark.init()
    spark = SparkSession\
        .builder\
        .master('local[*]')\
        .appName('imbalanced_bin_class')\
        .getOrCreate()
    # SET DIRECTORIES
    dir_root = 'path_to/HomeCredit'
    dir_data = os.path.join(dir_root, 'data')
    dir_outp = os.path.join(dir_root, 'outputs')
    # EXECUTE
    main()