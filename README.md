# Flight-Operations-Maintenance-Optimization ✈️
This project focuses on analyzing and predicting flight delays using real-world flight data from U.S. domestic airlines. The aim is to explore delay patterns, uncover correlations, and build a simple predictive model based on departure delays.
📊 Objective
Identify key trends and delay patterns based on hour, month, and airline

Explore the relationship between departure and arrival delays

Build a Linear Regression model to estimate arrival delay based on departure delay

🧰 Tools & Technologies
Python (Pandas, Matplotlib, Seaborn, Scikit-learn)

Dataset: flights.csv (1 year of domestic flight records from the U.S.)

Jupyter Notebook / Thonny (local execution)

📈 Key Insights
Strong correlation (0.91) between departure and arrival delays

Peak delays occur during late afternoon and early evening

Airlines show variation in average delay times — some carriers are consistently more punctual

🤖 Model Summary
Model: Linear Regression

Input: Departure delay (dep_delay)

Output: Predicted arrival delay (arr_delay)

Regression Results:
🔹 Mean Absolute Error (MAE): 13.13 minutes

🔹 Root Mean Squared Error (RMSE): 18.07 minutes

🔹 R² Score: 0.83
flight-delay-analysis/
│
├── data/
│   └── flights.csv.zip
│
├── notebooks/
│   └── 01_data_analysis_and_visualization.ipynb
│   └── 02_linear_regression_model.ipynb
│
├── outputs/
│   └── delay_by_airline.png
│   └── scatter_dep_arr_delay.png
│   └── model_metrics.txt
│
├── README.md
└── requirements.txt
📌 How to Run
Download the ZIP file containing the dataset.

Unzip and place flights.csv in the data/ folder.

Run notebooks in order:

01_data_analysis_and_visualization.ipynb

02_linear_regression_model.ipynb

📚 Acknowledgements
Data source: U.S. Department of Transportation / OpenFlightData
