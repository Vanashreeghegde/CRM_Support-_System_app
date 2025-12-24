import streamlit as st
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CRM Customer Support System",
    layout="centered"
)

st.title("📩 CRM Customer Support Ticketing System")
st.markdown("---")

# -----------------------------
# Sidebar: Role Selection
# -----------------------------


st.image("https://crystalsoftwares.in/img/helpdesk2.gif", use_container_width=True)


role = st.sidebar.radio("Select your role:", ["Customer", "Support Team"])

# -----------------------------
# Load Trained Pipeline
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("final.pkl")  # your trained pipeline

model = load_model()

# -----------------------------
# Initialize session state for ticket history
# -----------------------------
if 'ticket_history' not in st.session_state:
    st.session_state.ticket_history = pd.DataFrame(
        columns=['Ticket', 'Predicted Category', 'Confidence', 'Timestamp']
    )

# -----------------------------
# Function: Smart Prediction
# -----------------------------
def smart_predict(user_text, model):
    text = user_text.lower()
    # Keyword-based overrides
    if any(k in text for k in ["request", "need access", "please provide"]):
        return "Request"
    elif any(k in text for k in ["down", "outage", "not working", "error"]):
        return "Incident"
    else:
        return model.predict([user_text])[0]

# -----------------------------
# CUSTOMER VIEW
# -----------------------------
if role == "Customer":
    # Name input
    name = st.text_input("Enter your name:", max_chars=40)
    if st.button("Done"):
        if name.strip()=="":
            st.error("Kindly enter the name")
        else:
            st.success(f"Thank you for contacting us {name}😊")

    # Ticket input
    user_text = st.text_area(
        "Describe your issue:",
        placeholder="Example: My account is blocked and I cannot login",
        key="cust_ticket"
    )

    # Submit ticket
    if st.button("Submit Ticket", key="cust_submit"):
        if user_text.strip() == "":
            st.warning("Please enter a ticket description.")
        else:
            st.success(f"✅ Your ticket has been submitted successfully, {name} 👍")
            st.info("Our support team will contact you shortly.")

# -----------------------------
# SUPPORT TEAM VIEW
# -----------------------------
elif role == "Support Team":
    st.subheader("🧑‍💼 Support Dashboard")
    
    user_text = st.text_area(
        "Enter ticket text for classification:",
        placeholder="Type or paste ticket here..."
    )
    
    if st.button("Classify Ticket"):
        if user_text.strip() == "":
            st.warning("Please enter a ticket description.")
        else:
            # Predict probabilities for all classes
            probs = model.predict_proba([user_text])[0]
            
            # Get top 3 predictions
            top_n = 3
            top_indices = np.argsort(probs)[::-1][:top_n]
            top_classes = model.classes_[top_indices]
            top_probs = probs[top_indices] * 100
            
            # Add highest probability prediction to session history with timestamp
            st.session_state.ticket_history.loc[len(st.session_state.ticket_history)] = {
                'Ticket': user_text,
                'Predicted Category': top_classes[0],
                'Confidence': f"{top_probs[0]:.2f}%",
                'Timestamp': datetime.now()
            }
            
            # Display top 3 predictions
            st.success("✅ Ticket Classified Successfully")
            st.write("**Top Predictions:**")
            for cls, prob in zip(top_classes, top_probs):
                st.write(f"- {cls}: {prob:.2f}%")
    
    st.markdown("---")
    
    # -----------------------------
    # Ticket History Table
    # -----------------------------
    st.subheader("📋 Recent Tickets")
    if not st.session_state.ticket_history.empty:
        st.dataframe(st.session_state.ticket_history[::-1])  # newest on top
    else:
        st.info("No tickets classified yet.")
    
    st.markdown("---")
    
    # -----------------------------
    # Side-by-Side Charts
    # -----------------------------
    st.subheader("📊 Ticket Distribution by Category")
    
    if not st.session_state.ticket_history.empty:
        category_counts = st.session_state.ticket_history['Predicted Category'].value_counts()
        
        col1, col2 = st.columns([2, 2])  # Pie chart wider than bar chart
        
        # Pie chart (Plotly)
        with col1:
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                color_discrete_sequence=px.colors.sequential.RdBu,
                title="Category Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Bar chart (Matplotlib)
        with col2:
            fig_bar, ax = plt.subplots(figsize=(6,6))
            ax.bar(category_counts.index, category_counts.values, color='skyblue')
            ax.set_ylabel("Count")
            ax.set_title("Category Count")
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)
    else:
        st.info("No ticket data yet to show charts.")


st.markdown("---")
st.caption("Built by Vanashree Hegde 👩🏻‍🦰")
