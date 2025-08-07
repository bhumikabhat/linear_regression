# Multiple Linear Regression on Housing Dataset

This project implements a **Multiple Linear Regression** model to predict housing prices using various features such as area, number of bedrooms, location factors, and more.

## 📁 Dataset
The dataset used is `Housing.csv`, which includes features like:
- Area
- Bedrooms
- Bathrooms
- Parking
- Presence of facilities (e.g., main road access, guestroom, basement)
- Furnishing status

## 🧰 Tools & Libraries
- **Pandas** for data manipulation  
- **Scikit-learn** for model building  
- **Matplotlib** and **Seaborn** for data visualization

## ⚙️ Steps to Run

1. Clone or download this repository.
2. Ensure `Housing.csv` is in the same directory.
3. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
python multiple_linear_regression.py
📊 Model Evaluation
Mean Absolute Error (MAE): Approx. ₹9.7 Lakhs

Mean Squared Error (MSE): ~1.75 Trillion

R² Score: ~0.65

📈 Output
Displays a scatter plot comparing actual vs predicted prices.

✅ Conclusion
This basic regression model explains around 65% of the price variation. You can improve it further by:

Feature engineering

Regularization techniques (e.g., Ridge, Lasso)

Trying more advanced algorithms (e.g., Random Forest, XGBoost)
