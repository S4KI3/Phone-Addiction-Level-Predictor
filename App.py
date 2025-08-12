import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- Load Model -----------------
with open("addiction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset (must have "Addiction_Level" column)
df = pd.read_csv("your_dataset.csv")  # Change to your actual dataset file

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Phone Addiction Level Predictor",
    page_icon="📱",
    layout="centered"
)

# ----------------- App Title -----------------
st.title("📱 Phone Addiction Level Predictor")
st.markdown("""
This app predicts a **person's phone addiction level** based on their daily habits like phone usage, sleep hours, and social media activity. 
The model is trained on real-world data.

**Addiction Level Scale**:
- **Low** : 0 – 5  
- **Moderate** : 5 – 8  
- **High** : 8 – 10  
""")

st.markdown("---")

# ----------------- Input Form -----------------
st.header("📋 Enter User Details")

age = st.number_input(
    "Age (years)",
    min_value=10,
    max_value=100,
    help="Enter the age of the person whose phone addiction you want to check."
)

usage = st.number_input(
    "Daily Usage Hours (hours/day)",
    min_value=0.0,
    max_value=12.0,
    help="Average number of hours spent using the phone daily."
)

sleep = st.number_input(
    "Sleep Hours (hours/day)",
    min_value=0.0,
    max_value=24.0,
    help="Average number of hours of sleep per day."
)

checks = st.number_input(
    "Phone Checks Per Day",
    min_value=0,
    max_value=150,
    help="How many times you check your phone in a day."
)

apps = st.number_input(
    "Apps Used Daily",
    min_value=0,
    max_value=20,
    help="Number of different apps you use in a day."
)

social = st.number_input(
    "Time on Social Media (hours/day)",
    min_value=0.0,
    max_value=24.0,
    help="Average daily time spent on social media apps."
)

gaming = st.number_input(
    "Time on Gaming (hours/day)",
    min_value=0.0,
    max_value=24.0,
    help="Average daily time spent on mobile games."
)

# ----------------- Prediction -----------------
if st.button("🔍 Predict Addiction Level"):
    features = np.array([[age, usage, sleep, checks, apps, social, gaming]])
    pred = model.predict(features)[0]

    # Category Logic
    if pred < 5:
        category = "🟢 Low Addiction"
        color = "green"
    elif pred < 8:
        category = "🟡 Moderate Addiction"
        color = "orange"
    else:
        category = "🔴 High Addiction"
        color = "red"

    st.markdown(f"### 📊 Predicted Addiction Level: **{pred:.2f}**")
    st.markdown(f"<h4 style='color:{color}'>{category}</h4>", unsafe_allow_html=True)

    # Percentage of people with equal or higher addiction in dataset
    percentage = (df["Addiction_Level"] >= pred).mean() * 100
    st.write(f"📈 **{percentage:.2f}%** of people in the dataset have addiction level equal to or higher than you.")

    # Plot Graph
    fig, ax = plt.subplots()
    ax.bar(["You or Higher", "Lower than You"], [percentage, 100 - percentage],
           color=["red", "green"])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Comparison with Others in Dataset")
    st.pyplot(fig)

    # Recommendations
    st.markdown("**💡 Recommendations:**")
    if category == "🟢 Low Addiction":
        st.write("✅ Keep up the good habits! Maintain balance between phone use and other activities.")
    elif category == "🟡 Moderate Addiction":
        st.write("⚠️ Consider reducing screen time and increasing offline activities.")
    else:
        st.write("🚨 High risk of phone addiction! Try setting app timers and taking regular breaks.")

# ----------------- Footer -----------------
st.markdown("---")
st.caption("Developed with ❤️ by Sakib")
