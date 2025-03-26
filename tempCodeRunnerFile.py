numerical_variables = df.select_dtypes(include=[np.number]).columns
for col in numerical_variables:
    plt.figure(figsize=(8, 6))
    df[col].hist(bins=20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Assuming the "month" column contains integers (e.g., 1 for January, 2 for February, etc.)
df["month_numerical"] = df["month"].astype(int)

# Plot sales against month
plt.figure(figsize=(10, 6))
plt.scatter(df["month_numerical"], df["sales"])
plt.title("Sales vs. Month")
plt.xlabel("Month (numerical)")
plt.ylabel("Sales")
plt.show()

# Linear regression for sales and month
X = df[["month_numerical"]]  # Feature
y = df["sales"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and fit the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="black", label="Actual Sales")
plt.plot(
    X_test,
    regressor.predict(X_test),
    color="blue",
    linewidth=3,
    label="Regression Line",
)
plt.title("Linear Regression: Sales vs. Month")
plt.xlabel("Month (numerical)")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Predict sales for December (assuming December corresponds to month 12)
future_month = 12
predicted_sales = regressor.predict([[future_month]])
print(f"Predicted sales for December: {predicted_sales[0]:.2f}")

# Linear regression for sales, customer_reviews, and promotion_level
X = df[["customer_reviews", "promotion_level"]]  # Features
y = df["sales"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and fit the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict sales based on the test set
y_pred = regressor.predict(X_test)

# Plot predicted vs. actual sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.title("Actual vs. Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.show()
