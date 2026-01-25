import streamlit as st
import pandas as pd
from openai import OpenAI

# Initialize client
client = OpenAI(api_key="SECRET_KEY")

# Function to load KYC data
@st.cache_data  # caches the data for faster reloads
def load_data(file_path="_main_kyc_dataset.csv"):
    df = pd.read_csv(file_path)
    return df


df = load_data()
print(df.head())


# ============================================================================================================================================================
# Streamlit UI
# ============================================================================================================================================================
# --- Custom CSS to override Streamlit theme ---
# Page configuration
st.set_page_config(page_title="AI Powered Relationship Manager", layout="wide")

st.markdown(
    """
    <style>
    /* Change primary color (buttons, sliders, etc.) */
    :root {
        --primary-color: #1E90FF; /* DodgerBlue */
    }

    /* Optional: adjust text selection color */
    ::selection {
        background: #1E90FF33;
    }

    /* Make buttons rounded and blue */
    .stButton>button {
        background-color: #1E90FF !important;
        color: white !important;
        border-radius: 10px !important;
        border: none;
    }

    .stButton>button:hover {
        background-color: #187bcd !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Title ---
st.title("AI Powered Relationship Manager")

st.sidebar.image("VP_Bank_Logo.png", use_container_width=True)

# --- Sidebar info ---
st.sidebar.markdown(
    """
    This app is an **AI-powered assistant for VP Bank Relationship Managers**. It allows you to interact with your KYC and portfolio data in a conversational way. You can ask questions about clients, their holdings, risk profile, and other financial attributes.
    
    It uses artificial data.
    
    ---
    
    ### Example Questions

    **Lifestyle & Events**
    - I have free tickets to a boat show in Croatia. Which luxury lifestyle clients should I invite?
    - Which clients are likely to attend high-end art or cultural events?

    **Portfolio Insights**
    - Which clients have high exposure to tech stocks and aggressive risk profiles?
    - Show me clients with large alternative asset allocations.

    **New Product Offerings**
    - Identify clients likely to invest in sustainable or ESG funds.
    - Which clients might be interested in a new high-risk, high-return fund?
    - Find clients suitable for a new tech-focused mutual fund.
    - Which clients could benefit from structured investment products?     
    """
)

# --- Main page filters ---
st.markdown("### Filter Client Base")

col1, col2 = st.columns(2)

with col1:
    selected_segments = st.multiselect(
        "PC Segment:", options=df["pc_segment"].unique(), default=df["pc_segment"].unique()
    )

    selected_bu = st.multiselect(
        "BU Name", options=df["bu_name"].unique(), default=df["bu_name"].unique()
    )

    selected_nationality = st.multiselect(
        "Nationality", options=df["Nationality"].unique(), default=df["Nationality"].unique()
    )

with col2:
    selected_AML_risk = st.multiselect(
        "AML Risk", options=df["AML Risk"].unique(), default=df["AML Risk"].unique()
    )

    selected_RM = st.multiselect(
        "RM", options=df["RM"].unique(), default=df["RM"].unique()
    )

    selected_Affluence = st.multiselect(
        "Affluence", options=df["Affluence"].unique(), default=df["Affluence"].unique()
    )

    # --- Apply filters ---
df_slice = df[
    (df["bu_name"].isin(selected_bu)) &
    (df["pc_segment"].isin(selected_segments)) &
    (df["Nationality"].isin(selected_nationality)) &
    (df["AML Risk"].isin(selected_AML_risk)) &
    (df["RM"].isin(selected_RM)) &
    (df["Affluence"].isin(selected_Affluence))
]


# ============================================================================================================================================================
# User input for GPT Q&A
# ============================================================================================================================================================

st.markdown("""
    
    ---
                
    ## Ask AI Agent Q&A
    """
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content":
         "You are an elite data science assistant helping a VP Bank Relationship Manager. "
         "You have access to their client database. Allways returns client names and elaborate why him or her!"
         "Be conversational, remember previous questions, and react naturally."
         "Find answer in data, use different columns. Return mainly names, elaborate with short text."}
    ]

# Input field
question = st.chat_input("Ask a question about Clients...")

if question:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.spinner("🤖 AI is analyzing client data..."):

        # Add schema context once per run
        schema_context = f"""
        The filtered DataFrame has shape {df_slice.shape}.
        Columns: {list(df_slice.columns)}.
        Dtypes: {df_slice.dtypes.to_dict()}.
        Here are rows:
        {df_slice}
        """
        # Put schema into system if not yet added
        if not any("DataFrame" in m["content"] for m in st.session_state["messages"]):
            st.session_state["messages"].insert(1, {"role": "system", "content": schema_context})

        # Call GPT with full conversation
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=st.session_state["messages"]
        )

        answer = response.choices[0].message.content

        # Add assistant reply
        st.session_state["messages"].append({"role": "assistant", "content": answer})

# --- Render chat history ---
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

