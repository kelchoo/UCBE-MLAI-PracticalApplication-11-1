# PracticalApplication-11-1
 UC Berkeley Extension ML and AI - Practical application 11.1
## Business Understanding

From a business perspective, we are tasked with identifying key drivers for used car prices. In the CRISP-DM overview, we are asked to convert this business framing to a data problem definition. Using a few sentences, reframe the task as a data task with the appropriate technical vocabulary.

## Business Problem - understanding the difficulty of price prediction for used cars¶

The CRISP-DM process begins with the understanding of the business problem. Imagine for example a used car dealer who needs estimates what the price of a used car could be. The car dealer could be interest in predicting the price of a car based on its attributes. The need to answer business questions:

· Is the price of a car related to its features?

· Is the price of a care related to the condition of the car and its time since its manufactured date?

· Can the price of a car be predicted based in its attribute with reasonable accuracy?
Due to the increased price of new cars and the incapability of customers to buy new cars due to the lack of funds, used cars sales are on a global increase. Predicting the prices of used cars is an interesting and much-needed problem to be addressed. Customers can be widely exploited by fixing unrealistic prices for the used cars and many falls into this trap. Therefore, rises an absolute necessity of a used car price prediction system to effectively determine the worthiness of the car using a variety of features. Due to the adverse pricing of cars and the nomadic nature of people in developed countries, the cars are mostly bought on a lease basis, where there is an agreement between the buyer and seller. These cars upon completion of the agreement are resold. So reselling has become an essential part of today’s world.

Given the description of used cars, the prediction of used cars is not an easy task. There are a variety of features of a car like: the age of the car, its model, the manufacturer of the car (the original country of the manufacturer),its mileage or odometer reading in this case(the number of mildes it has run) Due to rising fuel prices because of the Ukraine/Russia war, fuel economy is also of prime importance - therefore fuel ('gas' 'diesel' 'hybrid' 'electric' 'other') are key determinants of price.

Tangible features: type of fuel it uses, style, braking system - rwd, awd, the volume of its cylinders (measured in cc), safety index, size,paint color, Intangibles features : consumer reviews, prestigious awards won by the car manufacturer. Therefore, understanding the key features driving price of used car is key to help both the dealership and the consumer arrive at the best approach to agreeing on a closing price.
As such, there arises a need for a model that can assign a price for a vehicle by evaluating its features taking the prices of other cars into consideration. In this Notebook, we use supervised learning methods to predict the prices of used cars. The model has been chosen after careful exploratory data analysis (linear regression and multiple regression) to determine the impact of each feature on price.
Methodology:
So, we propose a methodology using Machine Learning models to predict the prices of used cars given the features. The price is estimated based on the number of features as mentioned above.
#### Notebook outline
####  Step 1, we collect the data about used cars, identify important features that reflect the price.
#### Step 2, we preprocess and remove entries with NA values. Discard features that are not relevant for the prediction of the price.
#### Step 3, we apply ML models on the preprocessed dataset with features as inputs and the price as output.


## Data Understanding

After considering the business understanding, we want to get familiar with our data. 

1. read the csv file
2. review the data types and null value columns

#### Normalizing the data with cleansing 
4. normalize the data by filling the 0s in the price with mean price
5. normalize the year with mean(year) for the rows with nulls
6. normalize the odometer with mean(odometer) for the 1965 rows filled.
7. Set null VINs to NoVIN to preserve rows that have meaningful values
df["VIN"].fillna("No VIN", inplace = True)
8. Drop the rows with null values in the columns i.e. df2 = df.dropna(how='any',axis=0) 
9. creating a new key attribute with year of the vehicle, creating the Age of each vehicle based on the the difference to the max year (of all vehicles in the fleet)

### Visualizations:
<img width="853" alt="image" src="https://user-images.githubusercontent.com/115063137/204688640-ecb46685-69ab-422b-ab03-846b49c13ff1.png">
<img width="1338" alt="image" src="https://user-images.githubusercontent.com/115063137/204689316-7268cc7c-f780-4815-942a-600c8c3fb5d5.png">

Correlation of the key attributes that are numeric help drive the relationship to price:
<img width="849" alt="image" src="https://user-images.githubusercontent.com/115063137/204689393-7928c380-27ea-4ab5-84f2-b2386ae3b913.png">

box plot of odometer and Age(years) of the used vehicles help show the breadth of data across the data set
<img width="805" alt="image" src="https://user-images.githubusercontent.com/115063137/204689570-c6b0e086-e967-46d3-a1c5-d706a582e065.png">

Average price of all cars by Age(years) show the outliers:
<img width="1492" alt="image" src="https://user-images.githubusercontent.com/115063137/204689844-439f958f-4246-464f-a36e-52e8d2cdb105.png">

Count of vehicles by age shows the distribution of the age to the vehicles with 9 years being the most common age of the vehicles.
<img width="902" alt="image" src="https://user-images.githubusercontent.com/115063137/204690384-151b70bd-aaa5-49dc-8664-2046aaaf4ce6.png">

The condition of the vehicles are also key attribute of and shows the quality of the fleet 
<img width="949" alt="image" src="https://user-images.githubusercontent.com/115063137/204690459-217476cf-bcd0-4f63-ab7c-2fb1545f2b02.png">

Price Ranking of fuel and type used vehicle help to determine the important attribute of fuel and type in influencing price
<img width="1350" alt="image" src="https://user-images.githubusercontent.com/115063137/204690679-f480939f-7298-4cde-a038-bd6a51d74355.png">

Encoding values are key to normalizing non numeric attributes for multi regression.
<img width="783" alt="image" src="https://user-images.githubusercontent.com/115063137/204690994-67f1a316-15c2-4722-8504-137b709f0282.png">

<img width="964" alt="image" src="https://user-images.githubusercontent.com/115063137/204691075-a457b3b3-b8ee-4c93-9dc4-df10ada522e9.png">

Looking at OLS Regression Results
<img width="1233" alt="image" src="https://user-images.githubusercontent.com/115063137/204691202-4291c1b2-933c-4e77-a79a-6950749ec586.png">


we will use Python’s statsmodels module to implement Ordinary Least Squares(OLS) method of linear regression.
Introduction : 
A linear regression model establishes the relation between a dependent variable(y) and at least one independent variable(x) as : 
\hat{y}=b_1x+b_0  
In OLS method, we have to choose the values of b_1  and b_0  such that, the total sum of squares of the difference between the calculated and observed values of y, is minimised. 
Formula for OLS:

<img width="1029" alt="image" src="https://user-images.githubusercontent.com/115063137/204691536-e94124d6-d768-4953-9f3f-3880405b8c0a.png">
 

statsmodels : provides classes and functions for the estimation of many different statistical models. 

<img width="573" alt="image" src="https://user-images.githubusercontent.com/115063137/204691624-c5dbcdab-ae1c-43dc-91ee-0eb00e417c48.png">

In statistics, the variance inflation factor (VIF) is the ratio (quotient) of the variance of estimating some parameter in a model that includes multiple other terms (parameters) by the variance of a model constructed using only one term.It quantifies the severity of multicollinearity in an ordinary least squares regression analysis. It provides an index that measures how much the variance (the square of the estimate's standard deviation) of an estimated regression coefficient is increased because of collinearity

## Modeling
#### 1) Fit & transform train data(X_train) with scaler
#### 2) Transform test data(X_test) with fitted scaler¶
#### 3) Check for P-value & VIF after scaling
Reviewing the following models- Linear, Lasso, RandomForestRegressor, Ridge

<img width="1720" alt="image" src="https://user-images.githubusercontent.com/115063137/204692685-460c7f53-f136-4ecd-a701-cccf35e85673.png">

## Evaluation
Now that we've settled on our models and findings, it is time to deliver the information to the client. You should organize your work as a basic report that details your primary findings. Keep in mind that your audience is a group of used car dealers interested in fine tuning their inventory.
Here are my recommendations to deploy the above model:

- Business Understanding (CRISP-DM) : objective:maixmize revenues daily and reduce inventory on the lot/increase unit sales.

1.Load the set of inventory available on the lot daily at the start of the day

2.focus on taking the outlier inventory that includes vehicles that are too old (largest number of years from mannufactured date i.e. Age(years) above, Consider selling these rare old inventory using more specific non-conventional methods like eBay bidding and auction houses due their rarity

- Consumer perspective of the choices and preference of attribute

3. build a questionaire online before user arrives at the used car dealership on the features.

4. Modeling -focus on population for certain manufacturer if that is the top preference, build the regression model of price to odometer for just that population of used vehicles for example, Toyota.we can then use the best model to forecast the predicted price for that manufacturer and then recommend using VIF the top features that would give us the reason for pricing the recommended models shown in the below example. The Average price would serve as a starting point for negotiation and offering the price to the consumer based on the population of prices set in the database.

5. Evaluation - the buyer would then evaluate with the used car dealership on the combination of features and models with further drill into similar charts and tables below.

<img width="1118" alt="image" src="https://user-images.githubusercontent.com/115063137/204693264-d8332a8c-38f3-41f6-a5f3-6876fe7c8952.png">

In conclusion the CRISP-DM model allows the dealership to use this process daily to refresh their analysis in the following ways:
1. load the new data daily after adding or removing new inventory from purchasing from third parties and sales of inventory respectively.
2. understand the state of the new data
3. perform regression to build the VIF model that provides the ranking of correlated variables as well as correlation changes in odometer, age of the vehicles
4. Allow them to evaluate if their sales have improved from reviewing the counts of vehicles available for sale and the graphing and visualization of price trends daily against attribute combinations and tailor results of inventory searches to consumer needs quickly when they arrive at the dealership.
