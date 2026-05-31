import os
import json
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import subprocess

# Set page config
st.set_page_config(
    page_title="Toxicity Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek card styling */
    .metric-card {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(4px);
        margin-bottom: 20px;
    }
    
    /* Badges */
    .toxic-badge {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.4);
    }
    .safe-badge {
        background: linear-gradient(135deg, #22c55e 0%, #15803d 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(34, 197, 94, 0.4);
    }
    
    /* Text highlighter */
    .highlighted-text {
        line-height: 1.8;
        font-size: 1.15em;
        background-color: rgba(15, 23, 42, 0.3);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    .highlight-toxic {
        background-color: rgba(239, 68, 68, 0.25);
        color: #f87171;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border-bottom: 2px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Helper function: Clean user text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load models and caches
@st.cache_resource
def load_keras_model(model_path):
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model from {model_path}: {e}")
            return None
    return None

@st.cache_resource
def load_bert_model():
    if os.path.exists("models/bert_model"):
        try:
            from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
            model = TFAutoModelForSequenceClassification.from_pretrained("models/bert_model")
            tokenizer = AutoTokenizer.from_pretrained("models/bert_model")
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading BERT model from models/bert_model: {e}")
            return None, None
    return None, None

# Load precalculated EDA results
def load_eda_results():
    path = "models/eda_results.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# Load evaluation metrics
def load_metrics():
    path = "models/metrics.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# Check file status
eda_data = load_eda_results()
metrics_data = load_metrics()
lstm_exists = os.path.exists("models/bi_lstm_model.keras")
gru_exists = os.path.exists("models/bi_gru_model.keras")
bert_exists = os.path.exists("models/bert_model")
setup_complete = eda_data is not None and metrics_data is not None and (lstm_exists or gru_exists or bert_exists)

# App Title & Header
st.title("🛡️ Online Content Moderation & Comment Toxicity Detection")
st.markdown("---")

# ----------------- Fallback Screen if Models are Not Trained -----------------
if not setup_complete:
    st.info("👋 Welcome! The models and exploratory data statistics are not initialized yet.")
    st.markdown("### First Time Setup")
    st.write("To make the application functional, we need to run the data analysis and train our deep learning models. This can be triggered directly below.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>1. Precompute Data Insights</h4>
            <p>Runs the exploratory data analysis script on <b>Dataset/train.csv</b>. Computes word frequencies, correlations, and length distributions.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Data Analysis (eda.py)", use_container_width=True):
            with st.spinner("Analyzing train.csv (approx. 10-15s)..."):
                proc = subprocess.run(["python", "eda.py"], capture_output=True, text=True)
                if proc.returncode == 0:
                    st.success("Exploratory Data Analysis completed!")
                    st.rerun()
                else:
                    st.error("Analysis failed!")
                    st.code(proc.stderr)
                    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>2. Train Deep Learning Models</h4>
            <p>Trains deep learning models (<b>Bidirectional LSTM</b>, <b>Bidirectional GRU</b>, or <b>BERT Transformer</b>) on a sample of the dataset and saves evaluation metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        model_to_train = st.selectbox("Model to Train", ["both", "lstm", "gru", "bert", "all"], help="'both' trains LSTM and GRU. 'all' trains LSTM, GRU, and BERT.")
        sample_size = st.selectbox("Training Sample Size (Select 10k or 20k for fast CPU training)", [10000, 20000, 50000, "all"])
        epochs = st.slider("Epochs", 1, 5, 2)
        
        if st.button("Start Deep Learning Training (train.py)", use_container_width=True):
            with st.spinner("Training models in background (approx 2-5 mins on CPU)... Logs will display below."):
                log_placeholder = st.empty()
                log_placeholder.text("Training initiated...")
                
                cmd = ["python", "train.py", "--sample_size", str(sample_size), "--epochs", str(epochs), "--model_type", model_to_train]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                
                if proc.returncode == 0:
                    st.success("Training completed successfully! Loading application...")
                    st.rerun()
                else:
                    st.error("Training failed!")
                    st.code(proc.stderr)
                    st.code(proc.stdout)
    st.stop()

# ----------------- Normal App Flow (Setup Complete) -----------------
# Sidebar options
st.sidebar.image("https://img.icons8.com/color/120/shield.png", width=60)
st.sidebar.markdown("### Dashboard Control Panel")

# Navigation
page = st.sidebar.radio(
    "Select Screen:",
    ["Real-Time Detector", "Bulk Classifier", "Data Insights", "Model Performance"]
)

# Model Selection
available_models = []
if lstm_exists: available_models.append("LSTM (Bidirectional)")
if gru_exists: available_models.append("GRU (Bidirectional)")
if bert_exists: available_models.append("BERT (Transformer)")
selected_model_name = st.sidebar.selectbox("Active Inference Model:", available_models)

bert_model = None
bert_tokenizer = None
if "BERT" in selected_model_name:
    bert_model, bert_tokenizer = load_bert_model()
    model = bert_model
else:
    model_path = "models/bi_lstm_model.keras" if "LSTM" in selected_model_name else "models/bi_gru_model.keras"
    model = load_keras_model(model_path)

# Toxicity Threshold Slider
threshold = st.sidebar.slider("Toxicity Decision Threshold:", 0.1, 0.9, 0.5, 0.05, 
                             help="Comments with a probability score above this threshold will be flagged as toxic.")

st.sidebar.markdown("---")
st.sidebar.info("This application is powered by TensorFlow and Streamlit, evaluating text inputs against 6 categories of toxic language in real-time.")

# Target label details
target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
label_descriptions = {
    'toxic': "Rude, disrespectful, or unreasonable comment likely to make people leave a discussion.",
    'severe_toxic': "Extremely hateful, aggressive, or highly offensive comments.",
    'obscene': "Vulgar, rude, or offensive language, including swear words.",
    'threat': "Statements expressing an intention to inflict pain, injury, or hostile actions.",
    'insult': "Insulting or derogatory statements targeting an individual or group.",
    'identity_hate': "Hate speech targeting identity (race, religion, gender, sexual orientation, disability)."
}

# ----------------- 1. Real-Time Detector -----------------
if page == "Real-Time Detector":
    st.subheader("🛡️ Real-Time Comment Toxicity Analyzer")
    st.write("Type a comment below to inspect its toxicity likelihood across six distinct categories.")
    
    # Input area
    user_comment = st.text_area("Enter Comment Text:", value="You are extremely stupid and I hate your work.", height=120)
    
    if st.button("Analyze Comment", type="primary") or user_comment:
        if not user_comment.strip():
            st.warning("Please enter some text first.")
        else:
            # Clean and predict
            cleaned = clean_text(user_comment)
            
            # Predict
            with st.spinner("Analyzing text..."):
                if "BERT" in selected_model_name:
                    if bert_model is not None and bert_tokenizer is not None:
                        inputs = bert_tokenizer(
                            [cleaned],
                            padding="max_length",
                            truncation=True,
                            max_length=150,
                            return_tensors="np"
                        )
                        out = bert_model.predict(
                            {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}, 
                            verbose=0
                        )
                        logits = out.logits if hasattr(out, 'logits') else (out['logits'] if isinstance(out, dict) and 'logits' in out else out)
                        pred_prob = tf.nn.sigmoid(logits).numpy()[0]
                    else:
                        pred_prob = np.zeros(6)
                else:
                    pred_prob = model.predict(np.array([cleaned], dtype=object), verbose=0)[0]
                
            # Classify
            is_toxic = any(p >= threshold for p in pred_prob)
            
            # Display results header
            col_left, col_right = st.columns([2, 3])
            
            with col_left:
                st.markdown("### Analysis Outcome")
                if is_toxic:
                    st.markdown('<div class="toxic-badge">⚠️ FLAGGED TOXIC</div>', unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top: 10px;'>This comment has been flagged because one or more categories exceeded the threshold of <b>{threshold:.2f}</b>.</p>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-badge">✅ SAFE / APPROVED</div>', unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top: 10px;'>This comment is within safe guidelines (all categories are below the threshold of <b>{threshold:.2f}</b>).</p>", unsafe_allow_html=True)
                
                # Word Highlighter
                st.markdown("#### Highlighted Toxic Words")
                # Highlight toxic words from our EDA's list of toxic words
                toxic_words = [item['word'] for item in eda_data.get('toxic_top_words', [])][:15]
                words = user_comment.split()
                highlighted_words = []
                for w in words:
                    w_clean = re.sub(r"[^a-zA-Z]", "", w).lower()
                    if w_clean in toxic_words:
                        highlighted_words.append(f'<span class="highlight-toxic">{w}</span>')
                    else:
                        highlighted_words.append(w)
                
                highlighted_html = " ".join(highlighted_words)
                st.markdown(f'<div class="highlighted-text">{highlighted_html}</div>', unsafe_allow_html=True)
                st.caption("Highlighting is based on the top toxic terms identified in the training corpus.")

            with col_right:
                st.markdown("### Toxicity Category Probabilities")
                
                # Plot probabilities
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=pred_prob * 100,
                    y=[c.replace('_', ' ').title() for c in target_cols],
                    orientation='h',
                    marker=dict(
                        color=[ '#ef4444' if p >= threshold else '#3b82f6' for p in pred_prob],
                        line=dict(color='rgba(0, 0, 0, 0.1)', width=1)
                    ),
                    text=[f"{p*100:.1f}%" for p in pred_prob],
                    textposition='auto',
                ))
                
                # Add threshold line
                fig.add_vline(x=threshold * 100, line_dash="dash", line_color="orange",
                             annotation_text=f"Threshold ({threshold*100:.0f}%)", 
                             annotation_position="bottom right")
                
                fig.update_layout(
                    xaxis=dict(title="Probability (%)", range=[0, 100]),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=0, r=20, t=20, b=20),
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#888')
                )
                st.plotly_chart(fig, use_container_width=True)

            # Details expansion
            with st.expander("Show Category Meanings"):
                for col in target_cols:
                    st.write(f"**{col.replace('_', ' ').title()}**: {label_descriptions[col]}")

# ----------------- 2. Bulk Classifier -----------------
elif page == "Bulk Classifier":
    st.subheader("📤 Bulk Comment Moderation")
    st.write("Upload a CSV file of comments to moderate in batch. Preview outputs and download predictions.")
    
    uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded CSV with {len(df_upload)} rows.")
        
        # Select Column
        text_column = st.selectbox("Select the column containing the comments:", df_upload.columns)
        
        if st.button("Moderate Uploaded Comments", type="primary"):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Clean and predict
            status_text.text("Preprocessing text inputs...")
            cleaned_texts = df_upload[text_column].astype(str).apply(clean_text).values
            
            status_text.text(f"Running inference using {selected_model_name}...")
            if "BERT" in selected_model_name:
                if bert_model is not None and bert_tokenizer is not None:
                    inputs = bert_tokenizer(
                        list(cleaned_texts),
                        padding="max_length",
                        truncation=True,
                        max_length=150,
                        return_tensors="np"
                    )
                    out = bert_model.predict(
                        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}, 
                        batch_size=64, 
                        verbose=0
                    )
                    logits = out.logits if hasattr(out, 'logits') else (out['logits'] if isinstance(out, dict) and 'logits' in out else out)
                    preds = tf.nn.sigmoid(logits).numpy()
                else:
                    preds = np.zeros((len(cleaned_texts), 6))
            else:
                # Run inference in batches of 128
                preds = model.predict(np.array(cleaned_texts, dtype=object), batch_size=128, verbose=0)
            
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            # Add predictions to dataframe
            for i, col in enumerate(target_cols):
                df_upload[f"{col}_prob"] = preds[:, i]
                df_upload[f"is_{col}"] = (preds[:, i] >= threshold).astype(int)
            
            # Flag overall toxic
            df_upload['flagged_toxic'] = df_upload[[f"is_{col}" for col in target_cols]].any(axis=1).astype(int)
            
            # Preview prediction results
            st.markdown("### Predictions Preview")
            st.dataframe(df_upload[[text_column, 'flagged_toxic'] + [f"{col}_prob" for col in target_cols]].head(50))
            
            # Statistics
            toxic_count = int(df_upload['flagged_toxic'].sum())
            total_count = len(df_upload)
            clean_count = total_count - toxic_count
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total Comments", total_count)
            with col_m2:
                st.metric("Flagged Toxic", toxic_count, delta=f"{toxic_count/total_count*100:.1f}% of total", delta_color="inverse")
            with col_m3:
                st.metric("Approved Clean", clean_count, delta=f"{clean_count/total_count*100:.1f}% of total")
                
            # Pie Chart
            fig_pie = px.pie(
                names=["Safe", "Toxic (Flagged)"],
                values=[clean_count, toxic_count],
                color=["Safe", "Toxic (Flagged)"],
                color_discrete_map={"Safe": "#22c55e", "Toxic (Flagged)": "#ef4444"},
                title="Moderation Decisions Breakdown"
            )
            fig_pie.update_layout(height=280, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Download predicted CSV
            csv_data = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Annotated CSV",
                data=csv_data,
                file_name="moderated_comments.csv",
                mime="text/csv",
                use_container_width=True
            )

# ----------------- 3. Data Insights (EDA) -----------------
elif page == "Data Insights":
    st.subheader("📊 Corpus Exploratory Data Insights")
    st.write("Understand the distribution, label overlaps, and semantic composition of the training dataset.")
    
    if eda_data is None:
        st.warning("Data insights not loaded. Please run the data analysis on the home page first.")
    else:
        # Overview stats
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Total Train Samples", f"{eda_data['total_comments']:,}")
        with col_s2:
            st.metric("Toxic Comments", f"{eda_data['label_counts']['toxic_any']['count']:,}", 
                      f"{eda_data['label_counts']['toxic_any']['percentage']:.2f}%")
        with col_s3:
            st.metric("Clean Comments", f"{eda_data['label_counts']['clean']['count']:,}", 
                      f"{eda_data['label_counts']['clean']['percentage']:.2f}%")
        with col_s4:
            st.metric("Toxicity Classes", "6 categories")
            
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### Distribution of Toxicity Subcategories")
            # Subcategory counts bar chart
            categories = eda_data['target_cols']
            counts = [eda_data['label_counts'][c]['count'] for c in categories]
            percentages = [eda_data['label_counts'][c]['percentage'] for c in categories]
            
            fig_bar = go.Figure(go.Bar(
                x=counts,
                y=[c.replace('_', ' ').title() for c in categories],
                orientation='h',
                marker=dict(color='#3b82f6'),
                text=[f"{pct:.2f}%" for pct in percentages],
                textposition='outside'
            ))
            fig_bar.update_layout(
                xaxis_title="Count",
                yaxis=dict(autorange="reversed"),
                height=320,
                margin=dict(l=100, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_g2:
            st.markdown("#### Label Correlation Heatmap")
            # Label correlations
            fig_heat = px.imshow(
                eda_data['correlation_matrix'],
                x=[c.replace('_', ' ').title() for c in categories],
                y=[c.replace('_', ' ').title() for c in categories],
                color_continuous_scale="Viridis",
                labels=dict(color="Correlation"),
                text_auto=True
            )
            fig_heat.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888')
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        col_g3, col_g4 = st.columns(2)
        
        with col_g3:
            st.markdown("#### Top Words in Toxic Comments")
            # Word frequencies in toxic comments
            toxic_words_df = pd.DataFrame(eda_data['toxic_top_words'][:15])
            fig_toxic_words = px.bar(
                toxic_words_df,
                x='count',
                y='word',
                orientation='h',
                color_discrete_sequence=['#ef4444'],
                labels={'count': 'Occurrences', 'word': 'Term'}
            )
            fig_toxic_words.update_layout(
                yaxis=dict(autorange="reversed"),
                height=350,
                margin=dict(l=100, r=20, t=10, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888')
            )
            st.plotly_chart(fig_toxic_words, use_container_width=True)
            
        with col_g4:
            st.markdown("#### Top Words in Clean Comments")
            # Word frequencies in clean comments
            clean_words_df = pd.DataFrame(eda_data['clean_top_words'][:15])
            fig_clean_words = px.bar(
                clean_words_df,
                x='count',
                y='word',
                orientation='h',
                color_discrete_sequence=['#22c55e'],
                labels={'count': 'Occurrences', 'word': 'Term'}
            )
            fig_clean_words.update_layout(
                yaxis=dict(autorange="reversed"),
                height=350,
                margin=dict(l=100, r=20, t=10, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888')
            )
            st.plotly_chart(fig_clean_words, use_container_width=True)
            
        # Character/Word lengths distributions
        st.markdown("#### Comment Word Count Distribution")
        hist_data = eda_data['word_histogram']
        bin_edges = hist_data['bin_edges']
        # Calculate centers
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        
        fig_hist = go.Figure(go.Bar(
            x=bin_centers,
            y=hist_data['counts'],
            marker=dict(color='#8b5cf6'),
            width=(bin_edges[1] - bin_edges[0]) * 0.9
        ))
        fig_hist.update_layout(
            xaxis_title="Word Count per Comment",
            yaxis_title="Frequency",
            height=280,
            margin=dict(l=50, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#888')
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ----------------- 4. Model Performance (Evaluation) -----------------
elif page == "Model Performance":
    st.subheader("📈 Model Training & Evaluation Metrics")
    st.write("Compare the architectures, training curves, and validation performance of LSTM, GRU, and BERT models.")
    
    if metrics_data is None:
        st.warning("Model evaluation metrics not loaded. Train models on home screen first.")
    else:
        # Check trained model list
        trained_models = list(metrics_data.keys())
        
        # Tabs for metrics
        tab1, tab2, tab3 = st.tabs(["Performance Summary", "Learning Curves", "Confusion Matrices"])
        
        with tab1:
            st.markdown("#### Architecture Comparison Table")
            
            # Build metrics dataframe
            comparison_rows = []
            for m in trained_models:
                m_avg = metrics_data[m]["metrics"]
                display_name = "BERT (Transformer)" if m == "bert" else f"Bidirectional {m.upper()}"
                comparison_rows.append({
                    "Model": display_name,
                    "Macro ROC-AUC": f"{m_avg['macro_avg']['roc_auc']:.4f}",
                    "Micro ROC-AUC": f"{m_avg['micro_avg']['roc_auc']:.4f}",
                    "Macro Precision": f"{m_avg['macro_avg']['precision']:.4f}",
                    "Macro Recall": f"{m_avg['macro_avg']['recall']:.4f}",
                    "Macro F1-Score": f"{m_avg['macro_avg']['f1_score']:.4f}",
                })
            df_compare = pd.DataFrame(comparison_rows)
            st.table(df_compare)
            
            # ROC AUC breakdown by subcategory
            st.markdown("#### ROC-AUC Score by Toxicity Subcategory")
            fig_auc_comp = go.Figure()
            for m in trained_models:
                classes = list(metrics_data[m]["metrics"]["per_class"].keys())
                aucs = [metrics_data[m]["metrics"]["per_class"][c]["roc_auc"] for c in classes]
                display_name = "BERT" if m == "bert" else f"Bi-{m.upper()}"
                fig_auc_comp.add_trace(go.Bar(
                    x=[c.replace('_', ' ').title() for c in classes],
                    y=aucs,
                    name=display_name,
                    text=[f"{a:.3f}" for a in aucs],
                    textposition='auto'
                ))
            fig_auc_comp.update_layout(
                yaxis=dict(title="ROC-AUC Score", range=[0.5, 1.0]),
                barmode='group',
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888')
            )
            st.plotly_chart(fig_auc_comp, use_container_width=True)
            
        with tab2:
            st.markdown("#### Training History curves")
            selected_plot_model = st.selectbox("Select Model to view history:", trained_models)
            history = metrics_data[selected_plot_model]["history"]
            epochs_list = list(range(1, len(history['loss']) + 1))
            
            col_lc1, col_lc2 = st.columns(2)
            
            with col_lc1:
                # Loss curves
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=epochs_list, y=history['loss'], name='Train Loss', line=dict(color='#f87171', width=3)))
                fig_loss.add_trace(go.Scatter(x=epochs_list, y=history['val_loss'], name='Val Loss', line=dict(color='#ef4444', width=3, dash='dash')))
                fig_loss.update_layout(
                    title="Training & Validation Loss",
                    xaxis=dict(title="Epoch", tickmode='linear', tick0=1, dtick=1),
                    yaxis=dict(title="Binary Crossentropy Loss"),
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#888')
                )
                st.plotly_chart(fig_loss, use_container_width=True)
                
            with col_lc2:
                # Accuracy curves
                fig_acc = go.Figure()
                acc_key = 'binary_accuracy' if 'binary_accuracy' in history else 'accuracy'
                val_acc_key = 'val_binary_accuracy' if 'val_binary_accuracy' in history else 'val_accuracy'
                fig_acc.add_trace(go.Scatter(x=epochs_list, y=history[acc_key], name='Train Accuracy', line=dict(color='#60a5fa', width=3)))
                fig_acc.add_trace(go.Scatter(x=epochs_list, y=history[val_acc_key], name='Val Accuracy', line=dict(color='#3b82f6', width=3, dash='dash')))
                fig_acc.update_layout(
                    title="Training & Validation Accuracy",
                    xaxis=dict(title="Epoch", tickmode='linear', tick0=1, dtick=1),
                    yaxis=dict(title="Accuracy"),
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#888')
                )
                st.plotly_chart(fig_acc, use_container_width=True)

        with tab3:
            st.markdown("#### Validation Confusion Matrices (Threshold = 0.5)")
            selected_cf_model = st.selectbox("Select Model for Confusion Matrix:", trained_models, key="cf_model_sel")
            selected_cf_class = st.selectbox("Select Toxicity Subcategory:", [c.replace('_', ' ').title() for c in target_cols])
            
            class_key = selected_cf_class.lower().replace(' ', '_')
            cf = metrics_data[selected_cf_model]["metrics"]["per_class"][class_key]["confusion_matrix"]
            
            # Build 2x2 confusion matrix array
            matrix_arr = [
                [cf["tn"], cf["fp"]],
                [cf["fn"], cf["tp"]]
            ]
            
            fig_cf = px.imshow(
                matrix_arr,
                x=["Predicted Safe", "Predicted Toxic"],
                y=["Actual Safe", "Actual Toxic"],
                color_continuous_scale="Blues",
                text_auto=True,
                labels=dict(color="SamplesCount")
            )
            fig_cf.update_layout(
                height=300,
                margin=dict(t=20, b=20, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888')
            )
            st.plotly_chart(fig_cf, use_container_width=True)
            
            # Print metrics summary for this specific class
            cls_metrics = metrics_data[selected_cf_model]["metrics"]["per_class"][class_key]
            col_cf_m1, col_cf_m2, col_cf_m3, col_cf_m4 = st.columns(4)
            with col_cf_m1:
                st.metric("ROC-AUC", f"{cls_metrics['roc_auc']:.4f}")
            with col_cf_m2:
                st.metric("Precision", f"{cls_metrics['precision']:.4f}")
            with col_cf_m3:
                st.metric("Recall", f"{cls_metrics['recall']:.4f}")
            with col_cf_m4:
                st.metric("F1-Score", f"{cls_metrics['f1_score']:.4f}")
