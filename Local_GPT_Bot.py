import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

st.title("TinyLLM Offline Chatbot")
st.write("Ask any question and get answers from the locally running TinyLLM model.")
st.write("Type 'Bye Bot' to stop the chat.")

model_path = r"C:\Users\devas\MyDesktop\Tinyllma"

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(model_path)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def generate_answer(question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip()
    return answer

user_input = st.text_input("Enter your question here:")

if user_input:
    if user_input.strip().lower() == "bye bot":
        st.write("Goodbye! Bot is shutting down.")
        st.session_state.chat_history = []
    else:
        answer = generate_answer(user_input.strip())
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", answer))

if st.session_state.chat_history:
    for speaker, text in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")
