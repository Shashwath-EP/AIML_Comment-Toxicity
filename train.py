import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, GRU, Dense, GlobalMaxPool1D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

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

def tokenize_for_bert(texts, tokenizer, max_len=128):
    tokenized = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np"
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

def build_model(model_type, vectorizer, max_tokens=20000, embedding_dim=128, max_len=150):
    inputs = tf.keras.Input(shape=(), dtype=tf.string)
    x = vectorizer(inputs)
    x = Embedding(input_dim=max_tokens, output_dim=embedding_dim)(x)
    
    if model_type == 'lstm':
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
    elif model_type == 'gru':
        x = Bidirectional(GRU(64, return_sequences=True))(x)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    x = GlobalMaxPool1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(6, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs, name=f"Bi_{model_type.upper()}_Toxicity")
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')]
    )
    return model

def evaluate_model(model, val_inputs, val_labels, target_cols, is_bert=False):
    print(f"Evaluating {model.name}...")
    # Predict probabilities
    if is_bert:
        out = model.predict(val_inputs, batch_size=64)
        if hasattr(out, 'logits'):
            logits = out.logits
        elif isinstance(out, dict) and 'logits' in out:
            logits = out['logits']
        else:
            logits = out
        predictions = tf.nn.sigmoid(logits).numpy()
    else:
        predictions = model.predict(val_inputs, batch_size=128)
    
    # Binary predictions (threshold 0.5)
    pred_binary = (predictions >= 0.5).astype(int)
    
    metrics = {
        "per_class": {},
        "macro_avg": {},
        "micro_avg": {}
    }
    
    # Per-class metrics
    for i, col in enumerate(target_cols):
        # ROC AUC
        try:
            if len(np.unique(val_labels[:, i])) > 1:
                auc = float(roc_auc_score(val_labels[:, i], predictions[:, i]))
            else:
                auc = 0.5
        except Exception:
            auc = 0.5
            
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels[:, i], pred_binary[:, i], average='binary', zero_division=0
        )
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(val_labels[:, i], pred_binary[:, i], labels=[0, 1]).ravel()
        
        metrics["per_class"][col] = {
            "roc_auc": auc,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }
        
    # Overall ROC AUC
    class_aucs = [metrics["per_class"][c]["roc_auc"] for c in target_cols]
    metrics["macro_avg"]["roc_auc"] = float(np.mean(class_aucs)) if class_aucs else 0.5
    try:
        metrics["micro_avg"]["roc_auc"] = float(roc_auc_score(val_labels, predictions, average='micro'))
    except Exception:
        metrics["micro_avg"]["roc_auc"] = 0.5
        
    # Micro/macro average precision, recall, f1
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(val_labels, pred_binary, average='macro', zero_division=0)
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(val_labels, pred_binary, average='micro', zero_division=0)
    
    metrics["macro_avg"]["precision"] = float(macro_p)
    metrics["macro_avg"]["recall"] = float(macro_r)
    metrics["macro_avg"]["f1_score"] = float(macro_f)
    
    metrics["micro_avg"]["precision"] = float(micro_p)
    metrics["micro_avg"]["recall"] = float(micro_r)
    metrics["micro_avg"]["f1_score"] = float(micro_f)
    
    return metrics

def train_and_evaluate(args):
    print("--------------------------------------------------")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    print("--------------------------------------------------")
    
    # Load dataset
    csv_path = args.dataset
    print(f"Loading training dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Sample dataset if requested (makes CPU training fast)
    if args.sample_size > 0 and args.sample_size < len(df):
        print(f"Sampling dataset to {args.sample_size} records for faster training on CPU...")
        # Make sure we sample proportionally for minority classes
        # Separate toxic (any label=1) and clean
        df['any_toxic'] = df[target_cols].any(axis=1)
        toxic_df = df[df['any_toxic'] == True]
        clean_df = df[df['any_toxic'] == False]
        
        # Keep original class balance ratio roughly
        toxic_ratio = len(toxic_df) / len(df)
        toxic_sample_size = int(args.sample_size * toxic_ratio)
        clean_sample_size = args.sample_size - toxic_sample_size
        
        toxic_sample = toxic_df.sample(n=min(toxic_sample_size, len(toxic_df)), random_state=42)
        clean_sample = clean_df.sample(n=min(clean_sample_size, len(clean_df)), random_state=42)
        
        df = pd.concat([toxic_sample, clean_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        df.drop(columns=['any_toxic'], inplace=True)
    
    print(f"Cleaning comments...")
    df['cleaned_text'] = df['comment_text'].apply(clean_text)
    
    # Filter empty texts
    df = df[df['cleaned_text'] != ""].reset_index(drop=True)
    
    X = df['cleaned_text'].to_numpy()
    y = df[target_cols].to_numpy()
    
    print(f"Dataset Size for Training: {len(X)} comments")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # Text vectorization layer
    print("Fitting text vectorizer...")
    vectorizer = TextVectorization(
        max_tokens=args.max_tokens,
        output_sequence_length=args.max_len,
        output_mode='int'
    )
    vectorizer.adapt(X_train)
    
    models_dir = args.output_dir
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    metrics_path = os.path.join(models_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                all_metrics = json.load(f)
        except Exception:
            all_metrics = {}
    else:
        all_metrics = {}
    
    # Run training for requested models
    if args.model_type == 'all':
        model_types = ['lstm', 'gru', 'bert']
    elif args.model_type == 'both':
        model_types = ['lstm', 'gru']
    else:
        model_types = [args.model_type]
    
    for m_type in model_types:
        if m_type == 'bert':
            print(f"\n==================== Training BERT Model ====================")
            from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
            print("Loading BERT tokenizer and pre-trained model...")
            bert_name = "prajjwal1/bert-tiny"
            tokenizer = AutoTokenizer.from_pretrained(bert_name)
            model = TFAutoModelForSequenceClassification.from_pretrained(
                bert_name,
                num_labels=6,
                problem_type="multi_label_classification",
                from_pt=True
            )
            
            print("Tokenizing train and validation sets for BERT...")
            train_inputs = tokenize_for_bert(X_train, tokenizer, args.max_len)
            val_inputs = tokenize_for_bert(X_val, tokenizer, args.max_len)
            
            import tf_keras as tfk
            optimizer = tfk.optimizers.Adam(learning_rate=5e-5)
            model.compile(
                optimizer=optimizer,
                loss=tfk.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tfk.metrics.BinaryAccuracy(name='binary_accuracy')]
            )
            model.summary()
            
            early_stop = tfk.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            history = model.fit(
                train_inputs, y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_data=(val_inputs, y_val),
                callbacks=[early_stop],
                verbose=1
            )
            
            bert_model_path = os.path.join(models_dir, "bert_model")
            print(f"Saving BERT model to {bert_model_path}...")
            model.save_pretrained(bert_model_path)
            tokenizer.save_pretrained(bert_model_path)
            
            eval_metrics = evaluate_model(model, val_inputs, y_val, target_cols, is_bert=True)
            
        else:
            print(f"\n==================== Training Bi-{m_type.upper()} Model ====================")
            model = build_model(m_type, vectorizer, args.max_tokens, args.embedding_dim, args.max_len)
            model.summary()
            
            # Train with early stopping to prevent overfitting
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            history = model.fit(
                X_train, y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                verbose=1
            )
            
            # Save model
            model_path = os.path.join(models_dir, f"bi_{m_type}_model.keras")
            print(f"Saving model to {model_path}...")
            model.save(model_path)
            
            # Evaluate
            eval_metrics = evaluate_model(model, X_val, y_val, target_cols, is_bert=False)
        
        # Save training history
        history_dict = {}
        for k, v in history.history.items():
            history_dict[k] = [float(val) for val in v]
            
        all_metrics[m_type] = {
            "history": history_dict,
            "metrics": eval_metrics
        }
        
    # Write metrics to JSON
    metrics_path = os.path.join(models_dir, "metrics.json")
    print(f"\nSaving final evaluation metrics to {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    print("Training process finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train comment toxicity detection models.")
    parser.add_argument("--dataset", type=str, default="Dataset/train.csv", help="Path to train.csv")
    parser.add_argument("--sample_size", type=str, default="20000", help="Number of samples to train on. Set to 'all' or negative for full dataset.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_tokens", type=int, default=20000, help="Vocabulary size")
    parser.add_argument("--max_len", type=int, default=150, help="Max sequence length")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--model_type", type=str, default="both", choices=["lstm", "gru", "both", "bert", "all"], help="Model type to train")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    
    args = parser.parse_args()
    
    # Handle sample size string parameter
    if args.sample_size.lower() == 'all':
        args.sample_size = -1
    else:
        try:
            args.sample_size = int(args.sample_size)
        except ValueError:
            print("Invalid sample_size argument. Defaulting to 20000.")
            args.sample_size = 20000
            
    train_and_evaluate(args)
