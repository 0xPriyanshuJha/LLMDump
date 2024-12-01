import streamlit as st
from phi.agent import Agent
from langchain_ollama.llms import OllamaLLM


st.set_page_config(
    page_title="AI Health & Fitness Planner",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0fff4;
        border: 1px solid #9ae6b4;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fffaf0;
        border: 1px solid #fbd38d;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

def display_dietary_plan(plan_content):
    with st.expander("📋 Your Personalized Dietary Plan", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🎯 Why this plan works")
            st.info(plan_content.get("why_this_plan_works", "Information not available"))
            st.markdown("### 🍽️ Meal Plan")
            st.write(plan_content.get("meal_plan", "Plan not available"))

        with col2:
            st.markdown("### ⚠️ Important Considerations")
            considerations = plan_content.get("important_considerations", "").split('\n')
            for consideration in considerations:
                if consideration.strip():
                    st.warning(consideration)

def display_fitness_plan(plan_content):
    with st.expander("💪 Your Personalized Fitness Plan", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🎯 Goals")
            st.success(plan_content.get("goals", "Goals not specified"))
            st.markdown("### 🏋️‍♂️ Exercise Routine")
            st.write(plan_content.get("routine", "Routine not available"))

        with col2:
            st.markdown("### 💡 Pro Tips")
            tips = plan_content.get("tips", "").split('\n')
            for tip in tips:
                if tip.strip():
                    st.info(tip)

def main():
    if 'dietary_plan' not in st.session_state:
        st.session_state.dietary_plan = {}
        st.session_state.fitness_plan = {}
        st.session_state.qa_pairs = []
        st.session_state.plans_generated = False

    st.title("🏋️‍♂️ AI Health & Fitness Planner")
    st.markdown("""
        <div style='background-color: #00008B; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        Get personalized dietary and fitness plans tailored to your goals and preferences.
        Our AI-powered system considers your unique profile to create the perfect plan for you.
        </div>
    """, unsafe_allow_html=True)

    # Initialize Llama locally
    try:
        model = OllamaLLM(model="Llama-3.1")
    except Exception as e:
        st.error(f"❌ Error initializing Llama model: {e}")
        return

    st.header("👤 Your Profile")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, step=1, help="Enter your age")
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.1)
        activity_level = st.selectbox(
            "Activity Level",
            options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
            help="Choose your typical activity level"
        )
        dietary_preferences = st.selectbox(
            "Dietary Preferences",
            options=["Vegetarian", "Keto", "Gluten Free", "Low Carb", "Dairy Free"],
            help="Select your dietary preference"
        )

    with col2:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, step=0.1)
        sex = st.selectbox("Sex", options=["Male", "Female", "Other"])
        fitness_goals = st.selectbox(
            "Fitness Goals",
            options=["Lose Weight", "Gain Muscle", "Endurance", "Stay Fit", "Strength Training"],
            help="What do you want to achieve?"
        )

    if st.button("🎯 Generate My Personalized Plan", use_container_width=True):
        with st.spinner("Creating your perfect health and fitness routine..."):
            try:
                dietary_agent = Agent(
                    name="Dietary Expert",
                    role="Provides personalized dietary recommendations",
                    model=model,
                    instructions=[
                        "Consider the user's input, including dietary restrictions and preferences.",
                        "Suggest a detailed meal plan for the day, including breakfast, lunch, dinner, and snacks.",
                        "Provide a brief explanation of why the plan is suited to the user's goals.",
                        "Focus on clarity, coherence, and quality of the recommendations.",
                    ]
                )

                fitness_agent = Agent(
                    name="Fitness Expert",
                    role="Provides personalized fitness recommendations",
                    model=model,
                    instructions=[
                        "Provide exercises tailored to the user's goals.",
                        "Include warm-up, main workout, and cool-down exercises.",
                        "Explain the benefits of each recommended exercise.",
                        "Ensure the plan is actionable and detailed.",
                    ]
                )

                user_profile = f"""
                Age: {age}
                Weight: {weight}kg
                Height: {height}cm
                Sex: {sex}
                Activity Level: {activity_level}
                Dietary Preferences: {dietary_preferences}
                Fitness Goals: {fitness_goals}
                """

                dietary_plan_response = dietary_agent.run(user_profile)
                fitness_plan_response = fitness_agent.run(user_profile)

                st.session_state.dietary_plan = {
                    "why_this_plan_works": "High Protein, Healthy Fats, Moderate Carbohydrates, and Caloric Balance",
                    "meal_plan": dietary_plan_response.content,
                    "important_considerations": """
                    - Hydration: Drink plenty of water throughout the day
                    - Electrolytes: Monitor sodium, potassium, and magnesium levels
                    - Fiber: Ensure adequate intake through vegetables and fruits
                    - Listen to your body: Adjust portion sizes as needed
                    """
                }
                st.session_state.fitness_plan = {
                    "goals": "Build strength, improve endurance, and maintain overall fitness",
                    "routine": fitness_plan_response.content,
                    "tips": """
                    - Track your progress regularly
                    - Allow proper rest between workouts
                    - Focus on proper form
                    - Stay consistent with your routine
                    """
                }
                st.session_state.plans_generated = True

                display_dietary_plan(st.session_state.dietary_plan)
                display_fitness_plan(st.session_state.fitness_plan)

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()