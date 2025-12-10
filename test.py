import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Load data
df = pd.read_csv(r"C:\Users\Ruban Jit R\Documents\student_data.csv")


print("✅ Dataset Loaded:")
print(df)

# 2. Separate input (X) and output (y)
X = df[["Hours_Studied"]]
y = df["Pass"]

# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# 4. Create ML model
model = LogisticRegression()

# 5. Train the model
model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully!")

# 6. Test accuracy
accuracy = model.score(X_test, y_test)
print("✅ Model Accuracy:", accuracy * 100, "%")

# 7. Take user input for prediction
hours = float(input("\nEnter number of hours you studied: "))

# 8. Predict
result = model.predict([[hours]])

# 9. Output
if result[0] == 1:
    print("🎉 Prediction: PASS ✅")
else:
    print("❌ Prediction: FAIL")
