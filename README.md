# classification-project

The repository contains the files for Daniel Northcutt and Jared Godar's Codeup project on clustering and modeling of zillow real estate data to predict error estimates

---

## About the Project

### Project Goals

The main goal of this project is to be able to accurately predict the Zillow Zestimate error.

This will be accomplished by using past using property data from transactions of single family homes in 2017 and clustering of data to build at least four models, evaluating the effectiveness of each model, and testing the best model on new data is has never seen.

The ability to accurately value a home is essential for both buyers and sellers. The ability of us to predict error in zestimates will allow us to determine the major drivers of error then improve our estimates accordingly. Having the most accurate estimates possible is at the core of our business. 

### Project Description

This project provides the opportunity to create and evaluate multiple predictive models as well as implement other essential parts of the data science pipeline including data cleaning, imputing nulls, and clustering data to look for trends in subgroups.

It will involve pulling relevant data from a SQL database; cleaning that data; splitting the data into training, validation, and test sets; scaling data; feature engineering; exploratory data analysis; clustering; modeling; model evaluation; model testing; and effectively communicating findings in written and oral formats.

A home is often the most expensive purchase one makes in their lifetime. Having a good handle on pricing is essential for both buyers and sellers. An accurate pricing model factoring in the properties of similar homes will allow for appropriate prices to be set as well as the ability to identify under and overvalued homes. By determining drivers of error in our models, we can develop strategies to improve the models.

---

### Initial Questions

- What are the main drivers of estimate error?
- What are the relative importance of the assorted drivers?
- What factors reduce error?
- What factors don't matter?
- Are there any other potentially useful features that can be engineered from the current data available?
- Are the relationships suggested by initial visualizations statistically significant?
- Is the data balanced or unbalanced?
- Are there null values or missing data that must be addressed?
- Are there any duplicates in the dataset?
- Which model feature is most important for this data and business case?
- Which model evaluation metrics are most sensitive to this primary feature?

---

# Data Dictionary
| Feature                    | Datatype               | Description                                                           |
|:---------------------------|:-----------------------|:----------------------------------------------------------------------|
bathroomcnt                  |          float64       | bathroom count
bedroomcnt                   |          float64       | bedroom count
calculatedfinishedsquarefeet |          float64       | calc finished square feet
fips                         |          float64       | fips (county)
latitude                     |          float64       | latitude
longitude                    |          float64       | longitude
lotsizesquarefeet            |          float64       | lot size square feet
regionidcity                 |          float64       | city region
regionidcounty               |           float64       | city county
regionidzip                  |           float64       | city zip
yearbuilt                    |           float64       | year built
structuretaxvaluedollarcnt   |           float64       | structure tax value count
taxvaluedollarcnt            |           float64       | tax value count
landtaxvaluedollarcnt        |           float64       | land value count
taxamount                    |           float64       | tax amount
county                       |            object       | county name
age                          |           float64       | age of home
age_bin                      |           float64       | age bins of homes
taxrate                      |           float64       | tax rate
acres                        |           float64       | acres
acres_bin                    |           float64       | acre bin of homes
sqft_bin                     |           float64       | squarefoot bin of homes
structure_dollar_per_sqft    |           float64       | structure dollar per sqft
structure_dollar_sqft_bin    |           float64       | structure dollar per sqft bins 
land_dollar_per_sqft         |           float64       | land dollar per sqft
lot_dollar_sqft_bin          |           float64       | lot dollar per sqft
bath_bed_ratio               |           float64       | bath to bed ratio
cola                         |             int64       | cola
logerror_bins                |          category       | bin of logerror
baseline                     |           float64       | baseline target to beat
scaled_latitude              |           float64       | scaled latitude
scaled_longitude             |           float64       | scaled longitude
scaled_bathroomcnt           |           float64       | scaled bathroom count
scaled_taxrate               |           float64       | scaled tax rate
scaled_bedroomcnt            |           float64       | scaled bedroom count
scaled_lotsizesquarefeet     |           float64       | scaled lot size sqft
scaled_age                   |           float64       | scaled age of homes
scaled_acres                 |           float64       | scaled acres
scaled_bath_bed_ratio        |           float64       | scaled bath bed ratio
scaled_calculatedfinishedsquarefeet|     float64       | scaled calc sqft
area_cluster                 |            object       | area cluster grouping
size_cluster                 |            object       | size cluster grouping
price_cluster                |            object       | price cluster grouping
tax_cluster                  |            object       | tax cluster grouping
area_cluster_la_newer        |             uint8       | la newer area cluster
area_cluster_la_older        |             uint8       | la older area cluster
area_cluster_northwest_costal|             uint8       | northwest costal area cluster
area_cluster_palmdale_landcaster |         uint8       | palmdale-landcaster area cluster
area_cluster_santa_clarita   |             uint8       | santa clarita area cluster
area_cluster_se_coast        |             uint8       | se coast area cluster
size_cluster_1250_to_1650    |             uint8       | size cluster
size_cluster_1300_to_2000    |             uint8       | size cluster 
size_cluster_1500_to_1900    |             uint8       | size cluster 
size_cluster_1500_to_2800    |             uint8       | size cluster 
size_cluster_2300_to_4400    |             uint8       | size cluster 
size_cluster_2900_to_4000    |             uint8       | size cluster 
size_cluster_900_to_1200     |             uint8       | size cluster 
price_cluster_144000_to_355000|            uint8       | price cluster 
price_cluster_34000_to_110000 |            uint8       | price cluster 
price_cluster_420000_to_870000|            uint8       | price cluster
price_cluster_45000_to_173000 |            uint8       | price cluster
price_cluster_69000_to_210000 |            uint8       | price cluster
tax_cluster_1000_to_3000      |            uint8       | tax cluster
tax_cluster_16000_to_22000    |            uint8       | tax cluster
tax_cluster_30000_to_40000    |            uint8       | tax cluster
tax_cluster_5000_to_6000      |            uint8       | tax cluster
tax_cluster_8500_to_12000     |            uint8       | tax cluster
logerror                      |          float64       | log error - target variable
</br>
</br>

---

### Steps to Reproduce

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md.
- [ ] Download the `wrangle4.py`, `explore.py`, and `modelipynb` files into your working directory.
- [ ] Add your own `env` file to your directory. (user, password, host).
- [ ] Run the `report_notebook2.ipynb` workbook.


---

### The Plan

![story map](clustering_story_map.jpg)

1. **Acquire, clean, prepare, and split the data:**
    - Pull from Zillow database.
    - Eliminate any unnecessary or redundant fields.
    - Engineer new, potentially informative features.
    - Search for null values and respond appropriately (delete, impute, etc.).
    - Deal with outliers.
    - Scale data appropriately.
    - Divide the data in to training, validation, and testing sets (~50-30-20 splits)
2. **Exploratory data analysis:**
    - Visualize pairwise relationships looking for correlation with home value.
    - Note any interesting correlations or other findings.
    - Test presumptive relationships for statistical significance.
    - Think of what features would be most useful for model.
    - Employ clustering to look for relationships between specific sub-groups.
    - Record any other interesting observations or findings.
    *NOTE: This data analysis will be limited to the training dataset*
3. **Model generation, assessment, and optimization:**
    - Establish baseline performance (mean model error, assuming error is normally distributed).
    - Generate a basic regression model using only strongest drivers.
    - Calculate evaluation metrics to assess quality of models (RMSE, R^2, and p as primary metrics).
    - Generate additional models incorporating other existing fields.
    - Use k-best and recursive feature selection to determine features.
    - Engineer additional features to use in other models.
    - Evaluate ensemble of better models on validation data to look for overfitting.
    - Select the highest performing model.
    - Test that model with the previously unused and unseen test data once and only once.
4. **Streamline presentation**
    - Take only the most relative information from the working along and create a succinct report that walks through the rationale, steps, code, and observations for the entire data science pipeline of acquiring, cleaning, preparing, modeling, evaluating, and testing our model.
    - Run the explore.py code to run data exploratory analysis of the data thru statistic tests and clustering visualizations
    - Run the model.py to run the 5 different OLS models for the train and validate datasets followed by running on the testing dataset
    - 
5. ** Next Steps/Recommendations:
    - Thoughtfully impute missing values from the dataset to give a richer training set to explore
    - Modeling can be tested on other linear regressors (LassoLars, TweedieRegressor, and Poylnomial regression     tuning hyperparameters)
    - Employ cross-validation techniques while tunning additional models

---

### Key Findings

- Most important factors:
    - By incorporating cluster data, our model improved baseline performance nearly 3%.
    - Clustering provided a greater understanding of our data and allowed modeling that beat the baseline.
    - Features showed very low correlation with the target variable.
    - Using scaling, binning, and clustering gave the data a foundation for a stronger model.
- Model performance
    - Model 5 of 22 features (geoclusters, size clusters, price clusters, taxvaluedollarcnt,           
    structuretaxvaluedollarcnt, landtaxvaluedollarcnt, taxamount) was the strongest performing model
    - Tested dataset performed 3% better than the baseline RMSE
    - All models performed were OLS



