# SalesPredictor: Time Series Sales Analysis & Forecasting Tool

A data analysis and machine learning project for analyzing historical sales data and predicting future sales using linear regression models and time series analysis.

## Project Overview

SalesPredictor analyzes sales data across multiple time periods and identifies key factors influencing sales performance. The project leverages regression modeling to understand relationships between time, customer reviews, promotion levels, and sales figures, ultimately enabling accurate sales forecasting for future periods.

## Key Features

- **Exploratory Data Analysis**: Visualizes distributions and relationships in sales data
- **Time Series Analysis**: Examines sales trends across months
- **Linear Regression Modeling**: Predicts sales based on time periods and other factors
- **Multivariate Analysis**: Incorporates customer reviews and promotion levels as predictors
- **Sales Forecasting**: Projects future sales values for upcoming months
- **Visualization**: Creates intuitive plots of trends and model performance

## Data Description

The analysis uses sales data (`sales_data.csv`) containing several key variables:

- **Month**: Time period of sales data
- **Sales**: Revenue or units sold per period
- **Customer Reviews**: Average review scores (typically 1-5 scale)
- **Promotion Level**: Intensity of marketing/promotional activity

## Analysis Methods

### Exploratory Data Analysis
- Descriptive statistics for all variables
- Histograms showing distribution of months, sales, customer reviews, and promotion levels

### Sales vs. Time Analysis
- Scatter plots examining the relationship between months and sales
- Linear regression to identify and quantify time-based trends

### Multivariate Regression
- Multiple linear regression incorporating customer reviews and promotion levels
- Model evaluation through predicted vs. actual sales comparisons

## Key Findings

The analysis reveals:

1. A clear positive trend in sales over time, suggesting steady business growth
2. Customer reviews and promotion levels significantly influence sales performance
3. The linear regression model effectively captures sales patterns, enabling reliable predictions
4. Sales forecasts for future periods (e.g., December) based on historical patterns
5. The ability to make custom predictions using specific customer review and promotion level values

## File Structure

- `project_4.py`: Main Python script containing the complete analysis
- `tempCodeRunnerFile.py`: Auxiliary script with partial analysis code
- `sales_data.csv`: Dataset containing sales information and predictors
- Generated visualizations:
  - Histograms of all variables
  - Sales vs. Month scatter plot
  - Linear regression plot
  - Actual vs. Predicted sales comparison

## How to Run

1. Clone the repository
   ```
   git clone https://github.com/yourusername/SalesPredictor.git
   cd SalesPredictor
   ```

2. Install required packages
   ```
   pip install pandas numpy matplotlib scikit-learn
   ```

3. Run the analysis script
   ```
   python project_4.py
   ```

## Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning modeling
- **Linear Regression**: Predictive modeling algorithm

## Prediction Capabilities

The system can predict sales based on:

1. **Time period alone**: Forecast sales for future months
2. **Customer reviews and promotion levels**: Predict sales for specific marketing scenarios
3. **Custom combinations**: Generate predictions for any combination of month, reviews, and promotion levels

## Future Work

- Incorporate seasonal adjustments for more accurate time-based predictions
- Implement more advanced models (Random Forest, XGBoost) for comparison
- Add confidence intervals to sales predictions
- Create an interactive dashboard for real-time forecasting
- Integrate with business intelligence platforms

## Author

Adithya Sriramoju

## Acknowledgments

- Business analytics community for inspiring sales prediction approaches
- The Python data science ecosystem for their powerful tools
