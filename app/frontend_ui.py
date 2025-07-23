import json
import httpx
import streamlit as st
import requests
#from fpdf import FPDF
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from env import import_my_env

import_my_env()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

def call_backend(user_inputs):
    try:
        with httpx.Client(timeout=60) as client:
            #response = client.post("http://127.0.0.1:8000/recommend", json=user_inputs)
            response = client.post("http://backend:8000/recommend", json=user_inputs)
            return response
    except httpx.RequestError as e:
        st.error(f"Request failed: {e}")
        return None
                    
def func(llm):
    # Submit button
    if st.button("📤 Get Stock Advice"):
        user_data = {
            "age": age,
            "monthly_income": monthly_income,
            "monthly_expenses": monthly_expenses,
            "monthly_investment": monthly_investment,
            "annual_extra_investment": annual_extra_investment,
            "current_savings": current_savings,
            "risk_percent": risk_percent,
            "years": years,
            "expected_returns_percent": expected_returns_percent,
            "num_dependents": num_dependents,
            "investment_type": investment_type.split()[0],  # remove emoji
            "investor_knowledge": investor_knowledge.split()[0],
            "interested_sectors": interested_sectors,
            "has_health_insurance": has_health_insurance,
            "has_emergency_fund": has_emergency_fund,
        }

        with st.spinner("Analyzing your financial profile. This may take a few moments..."):
            try:
                response = call_backend(user_data)
                if response.status_code == 200:
                    advice_data = response.json()
                    st.session_state["advice_data"] = advice_data
                    st.session_state["final_advice"] = advice_data.get("advice", "")
                    #st.success("✅ Advice received!")
                    
                    print(f"\n\n\n***********User profile***********\n")
                    print(advice_data.get("user_profile", ""))
                    print(f"\n\n\n***********Advice***********\n") 
                    print(st.session_state["final_advice"])

                else:
                    st.error(f"❌ Backend error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Request failed: {e}")

    if "advice_data" in st.session_state:
        st.markdown("### 📈 Your Stock Recommendation")
        with st.container():
            #st.markdown("#### Backend Advice")
            st.markdown(f"<div class=\"advice-box\">{st.session_state['advice_data'].get('advice', 'No advice available.')}</div>", unsafe_allow_html=True)

        #pdf_bytes = generate_advice_pdf(st.session_state["advice_data"])
        #st.download_button("📥 Download Advice as PDF", pdf_bytes, "advice_report.pdf", mime="application/pdf")
    
    # 💬 Chat interface below advice
    if "final_advice" in st.session_state:
        st.markdown("---")
        st.markdown("### 💬 Have questions about your stock advice? Let’s dive deeper.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        #llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

        user_msg = st.chat_input("Ask a question related to your stock advice...")

        if user_msg:
            full_prompt = f"""You are a helpful financial assistant.\n\n\
                            Here is the user's financial profile and backend-generated recommendation:\n\
                            {json.dumps(st.session_state["advice_data"], indent=2)}\n\nNow the user asks:\n\
                            {user_msg} \n\n\
                            Please provide a clear, factual, and helpful response.
                            If the question is not related to finance, politely reject it and instruct the user to ask only finance-related questions.
                            """
            with st.spinner("Analyzing your advice and crafting a smart response..."):
                try:
                    response = llm.invoke(full_prompt).content.strip()
                except Exception as e:
                    response = f"⚠️ Failed to fetch LLM response: {e}"

            st.session_state.chat_history.append((user_msg, response))

        if st.session_state.chat_history:
            st.markdown("#### Chat History")
            with st.container():
                st.markdown("<div class=\"chat-box\">", unsafe_allow_html=True)

                # 🔁 Reverse the chat history so latest is on top
                for msg, resp in reversed(st.session_state.chat_history):
                    st.markdown(f"<div class='chat-user'><b>User:</b> {msg}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-bot'><b>Assistant:</b> {resp}</div><hr>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

st.set_page_config(page_title="Financial Adviser", layout="wide")

st.markdown("""
<style>
.advice-box, .chat-box {
    max-height: 400px;
    overflow-y: auto;
    padding: 1rem;
    border: 1px solid #ccc;
    background-color: #fafafa;
    border-radius: 6px;
    margin-bottom: 1rem;
}
.chat-user {
    background-color: #e7f3ff;
    padding: 0.5rem;
    border-radius: 5px;
    margin-bottom: 0.2rem;
}
.chat-bot {
    background-color: #f0f0f0;
    padding: 0.5rem;
    border-radius: 5px;
    margin-bottom: 0.5rem;
}
.chat-box {
    max-height: 400px;
    overflow-y: auto;
    padding: 1rem;
    border: 1px solid #ccc;
    background-color: #fafafa;
    border-radius: 6px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


st.title("🤖 Intelligent 💰 Financial Adviser")

#st.markdown("Welcome! Please fill out your financial profile below:")
st.markdown("📈 **Welcome to Financial Adviser!** \n\
             We're excited to have you on board. Your journey to smarter, more personalized investing starts here. \n\
             🚀 We'll begin by gathering a few details about your financial information, risk tolerance, and interests. \n\
             This helps us tailor stock insights and strategies that *fit you perfectly*. \n\
             **Ready to unlock your investment potential? Let’s get started! 💼** \n")

# def generate_advice_pdf(advice_data):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="Stock Advice Summary", ln=True, align="C")
#     for k, v in advice_data.items():
#         text = f"{k}:\n{v if isinstance(v, str) else json.dumps(v, indent=2)}"
#         for line in text.split("\n"):
#             pdf.multi_cell(0, 10, line)
#     return pdf.output(dest="S").encode("latin1")

# Layout: left column for inputs, right column for advice + chat
col1, col2 = st.columns([1, 2])

# --- Inputs (Left Side) ---
with col1:

    # Define inputs
    st.markdown("### 👤 Basic Information")
    age = st.number_input("🎂 How young at heart are you?\n (Enter your age between 18 and 100)", min_value=18, max_value=100, value=25)

    st.markdown("### 💰 Income & Expenses")

    monthly_income = st.number_input("💸 What's your average monthly income in INR?\n",
                                     min_value=10000,
                                     step=1000,
                                     value=50000,
                                     help="(Include all sources – job, business, rent, etc.)")
    monthly_expenses = st.number_input("🧾 What do you typically spend in a month? (INR)\n",
                                       min_value=5000,
                                       step=1000,
                                       value=20000,
                                       help="(This includes rent, groceries, travel, etc.)")
    monthly_investment = st.number_input("📈 How much do you usually invest each month? (INR)\n",
                                         min_value=5000,
                                         step=1000,
                                         value=20000,
                                         help="(e.g. SIPs, stocks, mutual funds, etc.)")
    annual_extra_investment = st.number_input("🎁 Any yearly bonus or lump sum investments? (INR)\n",
                                              min_value=0,
                                              step=1000,
                                              help="(Think of things like Diwali bonuses, tax savings, etc.)")
    current_savings = st.number_input("🏦 What's your current total savings? (INR)\n",
                                      min_value=0,
                                      step=100,
                                      help="(This can include your bank balance, FDs, liquid funds, etc.)")
    risk_percent = st.slider("⚖️ How much risk are you comfortable with on a scale of 0 to 100%?",
                            min_value=0, max_value=100,
                            value=5, step=1,
                            help="Lower value → Less risky stocks to protect investment. Higher → open to volatility for more returns.")
    years = st.slider("📆 How long do you plan to invest for in years?", 
                      min_value=0, max_value=20,
                      value=1, step=1,
                      help="Longer duration allows more aggressive and growth-oriented investments.")
    expected_returns_percent = st.slider("📊 What annual return do you hope to achieve?", 
                                         min_value=0, max_value=20,
                                         value=10, step=1,
                                         help="High expected returns usually imply higher risks. Choose wisely.")
    num_dependents = st.number_input("👨‍👩‍👧‍👦 How many people financially depend on you?",
                                     min_value=0, max_value=10,
                                     help="(Think spouse, children, parents — enter a number 0–10)")

    st.markdown("### 📊 Investment Preferences")

    investment_type = st.selectbox(
        "How would you describe your investment style?\n",
        ["Aggressive", "Moderate", "Slow"],
        help="(**Aggressive**: Volatile stocks, startups\n\
               **Moderate**: Mix of growth and stability\n\
               **Slow**: Safe, steady assets)"
    )
    investor_knowledge = st.selectbox("📚 What’s your comfort level with investing?\n",
                                      ["Beginner", "Intermediate", "Expert"],
                                      help="(**Beginner**: New to investing\n"
                                      "**Intermediate**: Used platforms, basic knowledge\n"
                                      "**Expert**: Market-savvy, may even day-trade)")

    sector_list = [
            "Automobile and Auto Components", "Capital Goods", "Chemicals", "Construction",
            "Construction Materials", "Consumer Durables", "Consumer Services", "Diversified",
            "Fast Moving Consumer Goods", "Financial Services", "Forest Materials", "Healthcare",
            "Information Technology", "Media Entertainment & Publication", "Metals & Mining",
            "Oil Gas & Consumable Fuels", "Power", "Realty", "Services", "Telecommunication", "Textiles"
    ]
    interested_sectors = st.multiselect("🏭 Which sectors excite you the most?\nPick from key sectors or leave blank for no specific preference:", sector_list)

    st.markdown("### 🛡️ Financial Safety Nets")

    has_health_insurance = st.radio(
        "Do you have health insurance coverage?\n",
        ["yes", "no"],
        help="(Helps protect against unexpected medical costs.)"
    ) == "yes"

    has_emergency_fund = st.radio(
        "🚨 Do you have an emergency fund?\n",
        ["yes", "no"],
        help="(Should cover 3–6 months of expenses in case of emergencies.)"
    ) == "yes"

# --- Output + Chat (Right Side) ---
with col2:
    # ✅ Validation checks before backend call
    if monthly_expenses > monthly_income:
        st.error("Monthly expenses cannot exceed monthly income.")
    elif monthly_income - monthly_expenses <= 0:
        st.error("Disposable income must be positive.")
    elif monthly_investment > (monthly_income - monthly_expenses):
        st.error("Monthly investment cannot exceed disposable income.")
    else:
        func(llm)
