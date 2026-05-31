import os
import re
import json
import pandas as pd
import numpy as np
from collections import Counter

# Define a standard list of English stopwords to avoid NLTK download issues
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "u", "r", "ur", "would", "shall", "could", "might", "must", "get", "go", "make",
    "like", "one", "say", "know", "think", "see", "time", "people", "some", "take",
    "year", "good", "well", "look", "only", "also", "new", "use", "two", "give"
}

def clean_text_for_eda(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Remove punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text

def run_eda(csv_path="Dataset/train.csv", output_dir="models"):
    print("Starting Exploratory Data Analysis...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load data
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    total_comments = len(df)
    print(f"Loaded {total_comments} comments.")

    # Target columns
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Calculate counts and percentages
    label_counts = {}
    for col in target_cols:
        count = int(df[col].sum())
        pct = float(count / total_comments * 100)
        label_counts[col] = {"count": count, "percentage": pct}
        print(f" - {col}: {count} ({pct:.2f}%)")

    # Clean comments vs toxic comments
    df['any_toxicity'] = df[target_cols].any(axis=1)
    toxic_count = int(df['any_toxicity'].sum())
    clean_count = total_comments - toxic_count
    
    label_counts["clean"] = {"count": clean_count, "percentage": float(clean_count / total_comments * 100)}
    label_counts["toxic_any"] = {"count": toxic_count, "percentage": float(toxic_count / total_comments * 100)}
    print(f" - Clean comments: {clean_count} ({label_counts['clean']['percentage']:.2f}%)")
    print(f" - Toxic (any class) comments: {toxic_count} ({label_counts['toxic_any']['percentage']:.2f}%)")

    # Co-occurrence / Correlation matrix
    corr_matrix = df[target_cols].corr().round(4).values.tolist()
    
    # Text lengths
    print("Calculating comment length distributions...")
    df['char_length'] = df['comment_text'].apply(lambda x: len(str(x)))
    df['word_length'] = df['comment_text'].apply(lambda x: len(str(x).split()))
    
    char_stats = {
        "mean": float(df['char_length'].mean()),
        "std": float(df['char_length'].std()),
        "median": float(df['char_length'].median()),
        "max": int(df['char_length'].max())
    }
    
    word_stats = {
        "mean": float(df['word_length'].mean()),
        "std": float(df['word_length'].std()),
        "median": float(df['word_length'].median()),
        "max": int(df['word_length'].max())
    }

    # Bins for word lengths histogram
    word_hist, word_bins = np.histogram(df['word_length'], bins=50, range=(0, 300))
    word_hist_data = {
        "counts": [int(x) for x in word_hist],
        "bin_edges": [float(x) for x in word_bins]
    }

    # Word Frequency / Common words
    print("Calculating word frequencies (toxic vs. clean comments)...")
    # Take a sample of 20,000 toxic and 20,000 clean comments to make calculation fast
    toxic_sample = df[df['any_toxicity'] == True]['comment_text'].sample(n=min(20000, toxic_count), random_state=42)
    clean_sample = df[df['any_toxicity'] == False]['comment_text'].sample(n=min(20000, clean_count), random_state=42)

    def get_top_words(text_series, top_n=30):
        words = []
        for text in text_series:
            cleaned = clean_text_for_eda(text)
            tokens = cleaned.split()
            filtered_tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
            words.extend(filtered_tokens)
        
        counter = Counter(words)
        return [{"word": word, "count": count} for word, count in counter.most_common(top_n)]

    print(" - Processing toxic sample words...")
    toxic_top_words = get_top_words(toxic_sample)
    print(" - Processing clean sample words...")
    clean_top_words = get_top_words(clean_sample)

    # Save results to JSON
    results = {
        "total_comments": total_comments,
        "label_counts": label_counts,
        "target_cols": target_cols,
        "correlation_matrix": corr_matrix,
        "char_stats": char_stats,
        "word_stats": word_stats,
        "word_histogram": word_hist_data,
        "toxic_top_words": toxic_top_words,
        "clean_top_words": clean_top_words
    }

    output_path = os.path.join(output_dir, "eda_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"EDA completed successfully. Results saved to {output_path}")

if __name__ == "__main__":
    run_eda()
