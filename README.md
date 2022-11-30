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

Notebook outline:

Step 1, we collect the data about used cars, identify important features that reflect the price.
Step 2, we preprocess and remove entries with NA values. Discard features that are not relevant for the prediction of the price.
Step 3, we apply ML models on the preprocessed dataset with features as inputs and the price as output.
Data Understanding¶

After considering the business understanding, we want to get familiar with our data. Write down some steps that you would take to get to know the dataset and identify any quality issues within. Take time to get to know the dataset and explore what information it contains and how this could be used to inform your business understanding.
