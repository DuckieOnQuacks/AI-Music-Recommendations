import tkinter as tk
from tkinter import ttk
import threading
from neural_networks import *
from XGBoost import *

# Function to handle predictions
def start_prediction(model_function, result_var, genre_input, country_input):
    genre_str = genre_input.get().strip()
    country = country_input.get().strip()
    genres = [g.strip() for g in genre_str.split(",") if g.strip()]  # Split genres by comma and remove extra spaces

    print(genre_str)
    print(genres)
    print(country)
    if not genres or not country:
        result_var.set("Please enter both genre and country.")
        return

    result_var.set("Predicting...")
    def run_model():
        try:
            results = model_function(genres, country)
            result_var.set(results)
        except Exception as e:
            result_var.set(f"Error: {str(e)}")

    threading.Thread(target=run_model).start()

# GUI Setup
root = tk.Tk()
root.title("Upcoming Artist Success Prediction")

frame = ttk.Frame(root, padding=20, width=500, height=400)

frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Title Label
title_label = ttk.Label(frame, text="Success Prediction", font=("Helvetica", 16))
title_label.grid(row=0, column=0, columnspan=3, pady=10)

# Input Fields
ttk.Label(frame, text="Enter Genre(s):").grid(row=1, column=0, sticky=tk.W)
genre_input = ttk.Entry(frame, width=30)
genre_input.grid(row=1, column=1, sticky=tk.W)

ttk.Label(frame, text="Enter Country:").grid(row=2, column=0, sticky=tk.W)
country_input = ttk.Entry(frame, width=30)
country_input.grid(row=2, column=1, sticky=tk.W)

# Results Variables
lr_result = tk.StringVar(value="Awaiting input...")
nn_result = tk.StringVar(value="Awaiting input...")
xgb_result = tk.StringVar(value="Awaiting input...")

# Linear Regression
#ttk.Label(frame, text="Linear Regression").grid(row=3, column=0, sticky=tk.W)
#ttk.Label(frame, textvariable=lr_result).grid(row=3, column=1, sticky=tk.W)
#ttk.Button(frame, text="Predict", command=lambda: start_prediction(run_linear_regression, lr_result, genre_input, country_input)).grid(row=3, column=2, padx=5)

# Neural Network
ttk.Label(frame, text="Neural Network").grid(row=4, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=nn_result).grid(row=4, column=1, sticky=tk.W)
ttk.Button(frame, text="Predict", command=lambda: start_prediction(run_neural_network, nn_result, genre_input, country_input)).grid(row=4, column=2, padx=5)

# XGBoost
ttk.Label(frame, text="XGBoost").grid(row=5, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=xgb_result).grid(row=5, column=1, sticky=tk.W)
ttk.Button(frame, text="Predict", command=lambda: start_prediction(run_xgboost, xgb_result, genre_input, country_input)).grid(row=5, column=2, padx=5)

# Exit Button
exit_button = ttk.Button(frame, text="Exit", command=root.destroy)
exit_button.grid(row=6, column=0, columnspan=3, pady=10)

# Run the Application
root.mainloop()
