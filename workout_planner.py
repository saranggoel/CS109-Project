import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("Personalized Workout Planner Using Beta Distributions")

# **Explanation Section**
st.subheader("Welcome to the Personalized Workout Planner!")
st.markdown("""
This app simulates a personalized workout plan and tracks your progress over time using probabilistic modeling.

- **What does it do?**
    - Dynamically adjusts the weight you lift based on your performance in previous sessions.
    - Tracks your success and adherence to the workout plan using a probability model (Beta distribution).
    - Visualizes your improvement in two key metrics:
        1. **Weight Progression**: See how the weight you lift changes over time.
        2. **Success Probability**: Track the likelihood of completing future workouts based on past performance.

- **What can you customize?**
    - Adherence Probability: How likely you are to complete each session.
    - Starting Workout Weight: The initial weight you can lift.
    - Duration: The number of weeks and sessions per week.
    - Weight Increase Factor: How aggressively weights increase after successes.
            \nNote: Some standard weight increase factors are 1-2% for bench press, 2-3% for squat, and 2-3% for deadlift.

Adjust the parameters using the sidebar and see how your progress evolves over time!
""")

# **User Inputs**
st.sidebar.header("Adjust Parameters")
adherence_prob = st.sidebar.slider("Adherence Probability", 0.1, 1.0, 0.8, step=0.1)
starting_difficulty = st.sidebar.number_input("Starting Workout Weight (lbs)", min_value=10, max_value=500, value=100)
num_weeks = st.sidebar.number_input("Number of Weeks", min_value=1, max_value=52, value=8)
sessions_per_week = st.sidebar.slider("Sessions Per Week", 1, 7, 3)
difficulty_increase_factor = st.sidebar.slider("Weight Increase Factor (%)", 0.1, 10.0, 5.0)

# Calculate total sessions
total_sessions = int(num_weeks * sessions_per_week)

# **1. Initialization**
alpha, beta = 1, 1  # Beta distribution parameters
current_weight = starting_difficulty  # Start with user-defined weight
workout_log = []  # To track progression

# **2. Simulate Workouts**
for session in range(total_sessions):
    # Simulate adherence (success if user completes the session)
    success = np.random.random() < adherence_prob
    
    # Update Beta distribution based on success or failure
    if success:
        alpha += 1
    else:
        beta += 1

    # Calculate mean success probability
    mean_success_probability = alpha / (alpha + beta)

    # Adjust weight based on success probability and adherence
    if success:
        current_weight *= (1 + (difficulty_increase_factor / 100.0) * mean_success_probability)
    else:
        current_weight *= (1 - (difficulty_increase_factor / 200.0) * (1 - mean_success_probability))
    
    # Ensure weight doesn't drop below the starting weight
    current_weight = max(current_weight, starting_difficulty)

    # Log the session
    workout_log.append({
        'Session': session + 1,
        'Success': success,
        'Alpha': alpha,
        'Beta': beta,
        'Weight': current_weight,
        'Mean_Success_Probability': mean_success_probability
    })

# **3. Convert Log to DataFrame**
workout_df = pd.DataFrame(workout_log)

# **4. Plot Final Beta Distribution**
st.subheader("Final Beta Distribution")
x = np.linspace(0, 1, 100)
y = beta_dist.pdf(x, alpha, beta)

fig, ax = plt.subplots()
ax.plot(x, y, label=f"Beta Distribution (Alpha={alpha}, Beta={beta})", color="orange")
ax.set_title("Final Beta Distribution of Success Probability")
ax.set_xlabel("Probability of Success")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)

# **5. Plot Weight Progress Over Time**
st.subheader("Weight Progress Over Time")
fig, ax = plt.subplots()
ax.plot(workout_df['Session'], workout_df['Weight'], label="Weight (lbs)", color="gold")
ax.set_title("Progression of Weight Over Sessions") 
ax.set_xlabel("Workout Session")
ax.set_ylabel("Weight (lbs)")
ax.legend()
st.pyplot(fig)


st.subheader("Mean Success Probability Over Time")
fig, ax = plt.subplots()
ax.plot(workout_df['Session'], workout_df['Mean_Success_Probability'], label="Mean Success Probability", color="blue")
ax.set_title("Mean Success Probability Over Workout Sessions")
ax.set_xlabel("Workout Session")
ax.set_ylabel("Mean Success Probability")
ax.set_ylim(0, 1)  # Limit y-axis to [0, 1]
ax.legend()
st.pyplot(fig)

# **6. Display Workout Log**
st.subheader("Workout Log")
workout_df["Mean Success Probability"] = workout_df["Mean_Success_Probability"]
workout_df.drop("Mean_Success_Probability", axis = 1, inplace=True)
st.dataframe(workout_df)