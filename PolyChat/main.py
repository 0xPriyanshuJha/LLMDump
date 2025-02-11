import streamlit as st
import requests

llms = {
    "Ollama": {"url": "http://localhost:11434/api/generate", "params": {"model": "llama3.1"}},
    "OpenAI": {"url": "https://api.openai.com/v1/chat/completions", "params": {"model": "gpt-4"}},
    "Mistral": {"url": "https://api.mistral.ai/v1/chat/completions", "params": {"model": "mistral-large"}},
    "Cohere": {"url": "https://api.cohere.ai/generate", "params": {"model": "command-r-plus"}},
    "Hugging Face": {"url": "https://api-inference.huggingface.co/models/facebook/opt-6.7b"}
}

st.set_page_config(page_title="Multi-LLM Chat", page_icon="🌐")
st.title("Chat with your choice of LLM")


#model selection
selected_model = st.selectbox("Choose your model", list(llms.keys()))
api_details = llms[selected_model]
api_url = api_details["url"]
st.session_state.setdefault("messages",[])


#for chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#user input
if user_input := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Prepare payload
    payload = {}
    if selected_model in ["OpenAI", "Mistral"]:
        payload = {"model": api_details["params"]["model"], "messages": [{"role": "user", "content": user_input}], "temperature": 0.7}
    elif selected_model == "Ollama":
        payload = {"model": api_details["params"]["model"], "prompt": user_input}
    elif selected_model == "Cohere":
        payload = {"model": api_details["params"]["model"], "prompt": user_input, "max_tokens": 300}
    elif selected_model == "Hugging Face":
        payload = {"inputs": user_input}
    
    headers = {}
    if selected_model == "OpenAI":
        headers = {"Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}", "Content-Type": "application/json"}
    elif selected_model == "Mistral":
        headers = {"Authorization": f"Bearer {st.secrets['MISTRAL_API_KEY']}", "Content-Type": "application/json"}
    elif selected_model == "Cohere":
        headers = {"Authorization": f"Bearer {st.secrets['COHERE_API_KEY']}", "Content-Type": "application/json"}
    elif selected_model == "Hugging Face":
        headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}", "Content-Type": "application/json"}
    
    # API Request
    response = requests.post(api_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        if selected_model in ["OpenAI", "Mistral"]:
            ai_response = response.json()["choices"][0]["message"]["content"]
        elif selected_model == "Ollama":
            ai_response = response.json()["response"]
        elif selected_model == "Cohere":
            ai_response = response.json()["generations"][0]["text"]
        elif selected_model == "Hugging Face":
            ai_response = response.json()[0]["generated_text"]
    else:
        ai_response = "Error: Unable to fetch response."
    
    with st.chat_message("assistant"):
        st.markdown(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
