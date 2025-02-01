from tkinter import *
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class AlgorithmDashboard:
    def __init__(self, root):
        self.window = root
        self.window.title("Heart Disease Prediction")
        self.window.geometry("500x350")
        
        # Main Title
        Label(self.window, text="Heart Disease Prediction", font=("Helvetica", 20, "bold")).pack(pady=20)
        
        # Subtitle
        Label(self.window, text="Logistic Regression Model", font=("Helvetica", 14)).pack(pady=10)
        
        # Button for Logistic Regression
        Button(self.window, text="Start Prediction", command=self.open_logistic_regression_page,
               font=("Helvetica", 14), width=20).pack(pady=20)

    def open_logistic_regression_page(self):
        LogisticRegressionPage(Toplevel(self.window))

class LogisticRegressionPage:
    def __init__(self, window):
        self.window = window
        self.window.title("Logistic Regression")
        self.window.geometry("500x400")
        
        Label(self.window, text="Logistic Regression (Heart Disease)", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Load dataset
        try:
            self.df = pd.read_csv("C:/Users/devis/Downloads/heart-disease.csv")
            self.selected_features = ["age", "chol", "thalach", "trestbps"]
            self.X = self.df[self.selected_features]
            self.y = self.df["target"]
            
            # Split dataset and train model
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy on the test set
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred) * 100  # Accuracy as a percentage
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset or train model: {e}")
            self.window.destroy()
            return
        
        # Input fields
        Label(self.window, text="Enter values for the following features:", font=("Helvetica", 12)).pack(pady=5)
        
        self.entries = {}
        for feature in self.selected_features:
            frame = Frame(self.window)
            frame.pack(pady=5)
            Label(frame, text=f"{feature}:", font=("Helvetica", 12), width=12, anchor="w").pack(side=LEFT)
            entry = Entry(frame, width=20)
            entry.pack(side=LEFT)
            self.entries[feature] = entry
        
        Button(self.window, text="Predict", command=self.run_logistic_regression).pack(pady=15)
        
        self.result_label = Label(self.window, text="", font=("Helvetica", 12), wraplength=450)
        self.result_label.pack(pady=10)

    def run_logistic_regression(self):
        try:
            # Collect input values and convert to float
            input_values = [[float(self.entries[feature].get()) for feature in self.selected_features]]
            input_df = pd.DataFrame(input_values, columns=self.selected_features)
            
            # Make prediction
            prediction = self.model.predict(input_df)
            result_text = f"Predicted Class: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}\n"
            result_text += f"Model Accuracy: {self.accuracy:.2f}%"
            
            # Update the result label
            self.result_label.config(text=result_text)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid Input: {e}")

# Main function
def main():
    root = Tk()
    app = AlgorithmDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()
