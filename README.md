# HOME CREDIT

### DESCRIPTION

End to end imbalanced classification problem using HomeCredit Kaggle dataset and Apache Spark ML

### ACHIEVEMENTS

* Data exploration, computation of class imbalance ratio
* Data cleansing: replacement of missing categorical and numerical variables
* Conversion of categorical features into one-hot vectors & merging numerical and onehot categorical features into single vector column
* Fitting a weighted logistic regression model to the training data
* Evaluating model on testing set 
* Ploting model performance (Receiver Operating Characteristics & confusion matrix) 
* Computing metrics for model evaluation (Precision, Recall, F1 score & area under ROC 


### CONSOLE OUTPUTS
```
LOADED DATA SET:

root
 |-- SK_ID_CURR: integer (nullable = true)
 |-- TARGET: integer (nullable = true)
 |-- NAME_CONTRACT_TYPE: string (nullable = true)
 |-- CODE_GENDER: string (nullable = true)
 |-- FLAG_OWN_CAR: string (nullable = true)
 |-- FLAG_OWN_REALTY: string (nullable = true)
 |-- CNT_CHILDREN: integer (nullable = true)
 |-- AMT_INCOME_TOTAL: double (nullable = true)
 |-- AMT_CREDIT: double (nullable = true)
 |-- AMT_ANNUITY: double (nullable = true)
 |-- AMT_GOODS_PRICE: double (nullable = true)
 |-- NAME_TYPE_SUITE: string (nullable = true)
 |-- NAME_INCOME_TYPE: string (nullable = true)
 |-- NAME_EDUCATION_TYPE: string (nullable = true)
 |-- NAME_FAMILY_STATUS: string (nullable = true)
 |-- NAME_HOUSING_TYPE: string (nullable = true)
 |-- REGION_POPULATION_RELATIVE: double (nullable = true)
 |-- DAYS_BIRTH: integer (nullable = true)
 |-- DAYS_EMPLOYED: integer (nullable = true)
 |-- DAYS_REGISTRATION: double (nullable = true)
 |-- DAYS_ID_PUBLISH: integer (nullable = true)
 |-- OWN_CAR_AGE: double (nullable = true)
 |-- FLAG_MOBIL: integer (nullable = true)
 |-- FLAG_EMP_PHONE: integer (nullable = true)
 |-- FLAG_WORK_PHONE: integer (nullable = true)
 |-- FLAG_CONT_MOBILE: integer (nullable = true)
 |-- FLAG_PHONE: integer (nullable = true)
 |-- FLAG_EMAIL: integer (nullable = true)
 |-- OCCUPATION_TYPE: string (nullable = true)
 |-- CNT_FAM_MEMBERS: double (nullable = true)
 |-- REGION_RATING_CLIENT: integer (nullable = true)
 |-- REGION_RATING_CLIENT_W_CITY: integer (nullable = true)
 |-- WEEKDAY_APPR_PROCESS_START: string (nullable = true)
 |-- HOUR_APPR_PROCESS_START: integer (nullable = true)
 |-- REG_REGION_NOT_LIVE_REGION: integer (nullable = true)
 |-- REG_REGION_NOT_WORK_REGION: integer (nullable = true)
 |-- LIVE_REGION_NOT_WORK_REGION: integer (nullable = true)
 |-- REG_CITY_NOT_LIVE_CITY: integer (nullable = true)
 |-- REG_CITY_NOT_WORK_CITY: integer (nullable = true)
 |-- LIVE_CITY_NOT_WORK_CITY: integer (nullable = true)
 |-- ORGANIZATION_TYPE: string (nullable = true)
 |-- EXT_SOURCE_1: double (nullable = true)
 |-- EXT_SOURCE_2: double (nullable = true)
 |-- EXT_SOURCE_3: double (nullable = true)
 |-- APARTMENTS_AVG: double (nullable = true)
 |-- BASEMENTAREA_AVG: double (nullable = true)
 |-- YEARS_BEGINEXPLUATATION_AVG: double (nullable = true)
 |-- YEARS_BUILD_AVG: double (nullable = true)
 |-- COMMONAREA_AVG: double (nullable = true)
 |-- ELEVATORS_AVG: double (nullable = true)
 |-- ENTRANCES_AVG: double (nullable = true)
 |-- FLOORSMAX_AVG: double (nullable = true)
 |-- FLOORSMIN_AVG: double (nullable = true)
 |-- LANDAREA_AVG: double (nullable = true)
 |-- LIVINGAPARTMENTS_AVG: double (nullable = true)
 |-- LIVINGAREA_AVG: double (nullable = true)
 |-- NONLIVINGAPARTMENTS_AVG: double (nullable = true)
 |-- NONLIVINGAREA_AVG: double (nullable = true)
 |-- APARTMENTS_MODE: double (nullable = true)
 |-- BASEMENTAREA_MODE: double (nullable = true)
 |-- YEARS_BEGINEXPLUATATION_MODE: double (nullable = true)
 |-- YEARS_BUILD_MODE: double (nullable = true)
 |-- COMMONAREA_MODE: double (nullable = true)
 |-- ELEVATORS_MODE: double (nullable = true)
 |-- ENTRANCES_MODE: double (nullable = true)
 |-- FLOORSMAX_MODE: double (nullable = true)
 |-- FLOORSMIN_MODE: double (nullable = true)
 |-- LANDAREA_MODE: double (nullable = true)
 |-- LIVINGAPARTMENTS_MODE: double (nullable = true)
 |-- LIVINGAREA_MODE: double (nullable = true)
 |-- NONLIVINGAPARTMENTS_MODE: double (nullable = true)
 |-- NONLIVINGAREA_MODE: double (nullable = true)
 |-- APARTMENTS_MEDI: double (nullable = true)
 |-- BASEMENTAREA_MEDI: double (nullable = true)
 |-- YEARS_BEGINEXPLUATATION_MEDI: double (nullable = true)
 |-- YEARS_BUILD_MEDI: double (nullable = true)
 |-- COMMONAREA_MEDI: double (nullable = true)
 |-- ELEVATORS_MEDI: double (nullable = true)
 |-- ENTRANCES_MEDI: double (nullable = true)
 |-- FLOORSMAX_MEDI: double (nullable = true)
 |-- FLOORSMIN_MEDI: double (nullable = true)
 |-- LANDAREA_MEDI: double (nullable = true)
 |-- LIVINGAPARTMENTS_MEDI: double (nullable = true)
 |-- LIVINGAREA_MEDI: double (nullable = true)
 |-- NONLIVINGAPARTMENTS_MEDI: double (nullable = true)
 |-- NONLIVINGAREA_MEDI: double (nullable = true)
 |-- FONDKAPREMONT_MODE: string (nullable = true)
 |-- HOUSETYPE_MODE: string (nullable = true)
 |-- TOTALAREA_MODE: double (nullable = true)
 |-- WALLSMATERIAL_MODE: string (nullable = true)
 |-- EMERGENCYSTATE_MODE: string (nullable = true)
 |-- OBS_30_CNT_SOCIAL_CIRCLE: double (nullable = true)
 |-- DEF_30_CNT_SOCIAL_CIRCLE: double (nullable = true)
 |-- OBS_60_CNT_SOCIAL_CIRCLE: double (nullable = true)
 |-- DEF_60_CNT_SOCIAL_CIRCLE: double (nullable = true)
 |-- DAYS_LAST_PHONE_CHANGE: double (nullable = true)
 |-- FLAG_DOCUMENT_2: integer (nullable = true)
 |-- FLAG_DOCUMENT_3: integer (nullable = true)
 |-- FLAG_DOCUMENT_4: integer (nullable = true)
 |-- FLAG_DOCUMENT_5: integer (nullable = true)
 |-- FLAG_DOCUMENT_6: integer (nullable = true)
 |-- FLAG_DOCUMENT_7: integer (nullable = true)
 |-- FLAG_DOCUMENT_8: integer (nullable = true)
 |-- FLAG_DOCUMENT_9: integer (nullable = true)
 |-- FLAG_DOCUMENT_10: integer (nullable = true)
 |-- FLAG_DOCUMENT_11: integer (nullable = true)
 |-- FLAG_DOCUMENT_12: integer (nullable = true)
 |-- FLAG_DOCUMENT_13: integer (nullable = true)
 |-- FLAG_DOCUMENT_14: integer (nullable = true)
 |-- FLAG_DOCUMENT_15: integer (nullable = true)
 |-- FLAG_DOCUMENT_16: integer (nullable = true)
 |-- FLAG_DOCUMENT_17: integer (nullable = true)
 |-- FLAG_DOCUMENT_18: integer (nullable = true)
 |-- FLAG_DOCUMENT_19: integer (nullable = true)
 |-- FLAG_DOCUMENT_20: integer (nullable = true)
 |-- FLAG_DOCUMENT_21: integer (nullable = true)
 |-- AMT_REQ_CREDIT_BUREAU_HOUR: double (nullable = true)
 |-- AMT_REQ_CREDIT_BUREAU_DAY: double (nullable = true)
 |-- AMT_REQ_CREDIT_BUREAU_WEEK: double (nullable = true)
 |-- AMT_REQ_CREDIT_BUREAU_MON: double (nullable = true)
 |-- AMT_REQ_CREDIT_BUREAU_QRT: double (nullable = true)
 |-- AMT_REQ_CREDIT_BUREAU_YEAR: double (nullable = true)


class ratio : 0.912

SUMMARY MISSING VARIABLES:

+------------------------------+-------------+-------+
| variable name                |   n missing |     % |
|------------------------------+-------------+-------|
| COMMONAREA_MEDI              |      214865 | 69.87 |
| COMMONAREA_AVG               |      214865 | 69.87 |
| COMMONAREA_MODE              |      214865 | 69.87 |
| NONLIVINGAPARTMENTS_MEDI     |      213514 | 69.43 |
| NONLIVINGAPARTMENTS_MODE     |      213514 | 69.43 |
| NONLIVINGAPARTMENTS_AVG      |      213514 | 69.43 |
| FONDKAPREMONT_MODE           |      210295 | 68.39 |
| LIVINGAPARTMENTS_MODE        |      210199 | 68.35 |
| LIVINGAPARTMENTS_MEDI        |      210199 | 68.35 |
| LIVINGAPARTMENTS_AVG         |      210199 | 68.35 |
| FLOORSMIN_MODE               |      208642 | 67.85 |
| FLOORSMIN_MEDI               |      208642 | 67.85 |
| FLOORSMIN_AVG                |      208642 | 67.85 |
| YEARS_BUILD_MODE             |      204488 | 66.5  |
| YEARS_BUILD_MEDI             |      204488 | 66.5  |
| YEARS_BUILD_AVG              |      204488 | 66.5  |
| OWN_CAR_AGE                  |      202929 | 65.99 |
| LANDAREA_AVG                 |      182590 | 59.38 |
| LANDAREA_MEDI                |      182590 | 59.38 |
| LANDAREA_MODE                |      182590 | 59.38 |
| BASEMENTAREA_MEDI            |      179943 | 58.52 |
| BASEMENTAREA_AVG             |      179943 | 58.52 |
| BASEMENTAREA_MODE            |      179943 | 58.52 |
| EXT_SOURCE_1                 |      173378 | 56.38 |
| NONLIVINGAREA_MEDI           |      169682 | 55.18 |
| NONLIVINGAREA_MODE           |      169682 | 55.18 |
| NONLIVINGAREA_AVG            |      169682 | 55.18 |
| ELEVATORS_MEDI               |      163891 | 53.3  |
| ELEVATORS_MODE               |      163891 | 53.3  |
| ELEVATORS_AVG                |      163891 | 53.3  |
| WALLSMATERIAL_MODE           |      156341 | 50.84 |
| APARTMENTS_MODE              |      156061 | 50.75 |
| APARTMENTS_MEDI              |      156061 | 50.75 |
| APARTMENTS_AVG               |      156061 | 50.75 |
| ENTRANCES_MODE               |      154828 | 50.35 |
| ENTRANCES_AVG                |      154828 | 50.35 |
| ENTRANCES_MEDI               |      154828 | 50.35 |
| LIVINGAREA_MEDI              |      154350 | 50.19 |
| LIVINGAREA_MODE              |      154350 | 50.19 |
| LIVINGAREA_AVG               |      154350 | 50.19 |
| HOUSETYPE_MODE               |      154297 | 50.18 |
| FLOORSMAX_MEDI               |      153020 | 49.76 |
| FLOORSMAX_AVG                |      153020 | 49.76 |
| FLOORSMAX_MODE               |      153020 | 49.76 |
| YEARS_BEGINEXPLUATATION_AVG  |      150007 | 48.78 |
| YEARS_BEGINEXPLUATATION_MEDI |      150007 | 48.78 |
| YEARS_BEGINEXPLUATATION_MODE |      150007 | 48.78 |
| TOTALAREA_MODE               |      148431 | 48.27 |
| EMERGENCYSTATE_MODE          |      145755 | 47.4  |
| OCCUPATION_TYPE              |       96391 | 31.35 |
| EXT_SOURCE_3                 |       60965 | 19.83 |
| AMT_REQ_CREDIT_BUREAU_WEEK   |       41519 | 13.5  |
| AMT_REQ_CREDIT_BUREAU_DAY    |       41519 | 13.5  |
| AMT_REQ_CREDIT_BUREAU_MON    |       41519 | 13.5  |
| AMT_REQ_CREDIT_BUREAU_QRT    |       41519 | 13.5  |
| AMT_REQ_CREDIT_BUREAU_HOUR   |       41519 | 13.5  |
| AMT_REQ_CREDIT_BUREAU_YEAR   |       41519 | 13.5  |
| NAME_TYPE_SUITE              |        1292 |  0.42 |
| DEF_30_CNT_SOCIAL_CIRCLE     |        1021 |  0.33 |
| OBS_60_CNT_SOCIAL_CIRCLE     |        1021 |  0.33 |
| DEF_60_CNT_SOCIAL_CIRCLE     |        1021 |  0.33 |
| OBS_30_CNT_SOCIAL_CIRCLE     |        1021 |  0.33 |
| EXT_SOURCE_2                 |         660 |  0.21 |
| AMT_GOODS_PRICE              |         278 |  0.09 |
| AMT_ANNUITY                  |          12 |  0    |
| CNT_FAM_MEMBERS              |           2 |  0    |
| DAYS_LAST_PHONE_CHANGE       |           1 |  0    |
+------------------------------+-------------+-------+ 

VARIABLE TYPES:

+-------------+---------+
| Features    |   count |
|-------------+---------|
| Categorical |      15 |
| Numerical   |     104 |
+-------------+---------+ 

FILLING MISSING CATEGORICAL VARIABLES:

+---------------------+-------------------------------+
| Name                | Replaced by (most frequent)   |
|---------------------+-------------------------------|
| NAME_TYPE_SUITE     | Unaccompanied                 |
| OCCUPATION_TYPE     | Laborers                      |
| FONDKAPREMONT_MODE  | reg oper account              |
| HOUSETYPE_MODE      | block of flats                |
| WALLSMATERIAL_MODE  | Panel                         |
| EMERGENCYSTATE_MODE | No                            |
+---------------------+-------------------------------+ 

FILLING MISSING NUMERICAL VARIABLES:

+------------------------------+---------------------+
| Name                         |   Replaced by (AVG) |
|------------------------------+---------------------|
| AMT_ANNUITY                  |               31689 |
| AMT_GOODS_PRICE              |              640656 |
| OWN_CAR_AGE                  |                  11 |
| CNT_FAM_MEMBERS              |                   2 |
| EXT_SOURCE_1                 |                   1 |
| EXT_SOURCE_2                 |                   1 |
| EXT_SOURCE_3                 |                   0 |
| APARTMENTS_AVG               |                   0 |
| BASEMENTAREA_AVG             |                   0 |
| YEARS_BEGINEXPLUATATION_AVG  |                   1 |
| YEARS_BUILD_AVG              |                   1 |
| COMMONAREA_AVG               |                   0 |
| ELEVATORS_AVG                |                   0 |
| ENTRANCES_AVG                |                   0 |
| FLOORSMAX_AVG                |                   0 |
| FLOORSMIN_AVG                |                   0 |
| LANDAREA_AVG                 |                   0 |
| LIVINGAPARTMENTS_AVG         |                   0 |
| LIVINGAREA_AVG               |                   0 |
| NONLIVINGAPARTMENTS_AVG      |                   0 |
| NONLIVINGAREA_AVG            |                   0 |
| APARTMENTS_MODE              |                   0 |
| BASEMENTAREA_MODE            |                   0 |
| YEARS_BEGINEXPLUATATION_MODE |                   1 |
| YEARS_BUILD_MODE             |                   1 |
| COMMONAREA_MODE              |                   0 |
| ELEVATORS_MODE               |                   0 |
| ENTRANCES_MODE               |                   0 |
| FLOORSMAX_MODE               |                   0 |
| FLOORSMIN_MODE               |                   0 |
| LANDAREA_MODE                |                   0 |
| LIVINGAPARTMENTS_MODE        |                   0 |
| LIVINGAREA_MODE              |                   0 |
| NONLIVINGAPARTMENTS_MODE     |                   0 |
| NONLIVINGAREA_MODE           |                   0 |
| APARTMENTS_MEDI              |                   0 |
| BASEMENTAREA_MEDI            |                   0 |
| YEARS_BEGINEXPLUATATION_MEDI |                   1 |
| YEARS_BUILD_MEDI             |                   1 |
| COMMONAREA_MEDI              |                   0 |
| ELEVATORS_MEDI               |                   0 |
| ENTRANCES_MEDI               |                   0 |
| FLOORSMAX_MEDI               |                   0 |
| FLOORSMIN_MEDI               |                   0 |
| LANDAREA_MEDI                |                   0 |
| LIVINGAPARTMENTS_MEDI        |                   0 |
| LIVINGAREA_MEDI              |                   0 |
| NONLIVINGAPARTMENTS_MEDI     |                   0 |
| NONLIVINGAREA_MEDI           |                   0 |
| TOTALAREA_MODE               |                   0 |
| OBS_30_CNT_SOCIAL_CIRCLE     |                   1 |
| DEF_30_CNT_SOCIAL_CIRCLE     |                   0 |
| OBS_60_CNT_SOCIAL_CIRCLE     |                   1 |
| DEF_60_CNT_SOCIAL_CIRCLE     |                   0 |
| DAYS_LAST_PHONE_CHANGE       |               -1105 |
| AMT_REQ_CREDIT_BUREAU_HOUR   |                   0 |
| AMT_REQ_CREDIT_BUREAU_DAY    |                   0 |
| AMT_REQ_CREDIT_BUREAU_WEEK   |                   0 |
| AMT_REQ_CREDIT_BUREAU_MON    |                   0 |
| AMT_REQ_CREDIT_BUREAU_QRT    |                   0 |
| AMT_REQ_CREDIT_BUREAU_YEAR   |                   2 |
+------------------------------+---------------------+ 

ADDING WEIGHT COLUMN:

+------+-------------------+
|TARGET|             WEIGHT|
+------+-------------------+
|     1| 0.9121817139865434|
|     0|0.08781828601345665|
|     0|0.08781828601345665|
|     0|0.08781828601345665|
|     0|0.08781828601345665|
+------+-------------------+
only showing top 5 rows

CONVERTING CATEGORICAL FEATURES TO ONEHOT VECTORS:

+--------------------+
|            FEATURES|
+--------------------+
|(227,[0,1,2,3,9,1...|
|(227,[0,1,4,12,17...|
|(227,[2,3,9,16,21...|
|(227,[0,1,2,3,9,1...|
|(227,[0,1,2,3,9,1...|
+--------------------+
only showing top 5 rows

FITTING LOGISTIC REGRESSION MODEL

EVALUATING MODEL: LOGISTICREGRESSION

Confusion matrix:
+----------+------------+------------+
| •        |   Positive |   Negative |
|----------+------------+------------|
| Positive |   0.762013 |   0.237987 |
| Negative |   0.222991 |   0.777009 |
+----------+------------+------------+ 

Metrics:
+----------------+----------+
| •              |    Value |
|----------------+----------|
| Precision      | 0.762013 |
| Recall         | 0.773614 |
| F1 score       | 0.76777  |
| Area under ROC | 0.85709  |
+----------------+----------+ 
```