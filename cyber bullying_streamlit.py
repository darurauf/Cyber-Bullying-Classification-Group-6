import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Custom CSS dengan tema biru monokromatik
st.markdown("""
    <style>
        /* Semua teks hitam */
        * {
            color: #000000 !important;
        }
        
        /* Main background */
        .stApp {
            background-color: #f8fafc;
        }
        
        /* Headers */
        h1 {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 2px solid #1e40af;
            padding-bottom: 10px;
        }
        
        /* Text area */
        .stTextArea textarea {
            background-color: #ffffff;
            border: 2px solid #1e40af;
            border-radius: 8px;
            padding: 10px;
        }
        
        /* Button */
        .stButton>button {
            background-color: #ceeafb;
            color: #ffffff  !important;
            border: none;
            padding: 3px 24px;
            text-align: center;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #a1d6f7;
            color: white !important;
        }
        
        /* Prediction result box */
        .stMarkdown {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
            padding: 6px;
            border-radius: 8px;
            background-color: #f8fafc;
            text-align: center;
        }
        
        /* Fakta section */
        .fact-box {
            background-color: #f8fafc;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border-left: 3px solid #37a8ef;
            border-right: 3px solid #37a8ef;
            text-align: justify;
            text-justify: inter-word;
        }
        
        .fact-title {
            color: #000000 !important;
            font-weight: bold;
            margin-bottom: 8px;
            text-align: left;
        }
        
        .fact-content {
            font-weight: normal !important;
        }
        
        /* Paragraph justification */
        div[data-testid="stMarkdownContainer"] p {
            text-align: justify;
            text-justify: inter-word;
            font-weight: normal;
        }
        
        /* Warning message */
        .stAlert {
            background-color: #8accf5 !important;
            text-align: center;
        }
        
        
        /* Footer */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0f75b5;
            color: white !important;
            text-align: center;
            padding: 10px;
        }
        
        /* Placeholder text */
        .stTextArea textarea::placeholder {
            color: #666666 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Muat model, vectorizer, dan label encoder
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Fungsi untuk prediksi
def predict(text):
    text_vectorized = vectorizer.transform([text])
    prediction_encoded = svm_model.predict(text_vectorized)
    prediction = label_encoder.inverse_transform(prediction_encoded)
    return prediction[0]

# Antarmuka Streamlit
st.title("Cyber Bullying Prediction Using SVM Model")

input_text = st.text_area("Enter text to predict:", 
                         placeholder="Type or paste text here...")
if st.button("Predict"):
    if input_text:
        result = predict(input_text)
        st.markdown(f'<div class="prediction-result">Type of Cyber Bullying: {result}</div>', 
                   unsafe_allow_html=True)
    else:
        st.warning("Please insert text to predict!")

# Bagian Fakta-fakta Cyberbullying
st.markdown("---")
st.markdown("<h2 style='text-align: center;'>Facts about Cyberbullying</h2>", unsafe_allow_html=True)

facts = [
    {
        "title": "Global Prevalence🌍",
        "content": "According to UNICEF, about 1 in 3 young people in 30 countries report having experienced cyberbullying. Cyberbullying is a global problem affecting children and adolescents in many parts of the world, with a significant impact on their mental health."
    },
    {
        "title": "Psychological Impacts💔",
        "content": "Victims of cyberbullying have a 2-3 times higher risk of experiencing depression and anxiety than non-victims. These psychological effects can manifest in the form of sleep disturbances, decreased academic performance, and thoughts of self-harm."
    },
    {
        "title": "Anonymity🎭",
        "content": "70% of cyberbullying perpetrators utilize the anonymity of the internet to hide their identity. Online anonymity gives perpetrators the courage to commit acts that they might not do in face-to-face interactions."
    },
    {
        "title": "Social Media📱",
        "content": "Social media platforms are the most common places where cyberbullying occurs, with Instagram being the highest (42%). Facebook, Twitter, and online gaming platforms are also places that are prone to digitally bullying behavior."
    },
    {
        "title": "Long-term Impact⏳",
        "content": "The effects of cyberbullying can last well into adulthood, affecting victims' personal and professional relationships. Many victims report difficulties in building confidence and interpersonal relationships even years after the incident."
    }
]

for fact in facts:
    st.markdown(
        f"""
        <div class="fact-box">
            <div class="fact-title">{fact['title']}</div>
            <div class="fact-content">{fact['content']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("""
    <div class="footer">
        Cyber Bullying Classification by Group 6
    </div>
""", unsafe_allow_html=True)