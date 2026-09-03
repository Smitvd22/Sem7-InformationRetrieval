import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import string

def setup_nltk():
    """Download necessary NLTK datasets quietly."""
    packages = [
        'punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 
        'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'
    ]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

def get_wordnet_pos(treebank_tag):
    """Convert POS tag to WordNet POS tag."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default to noun
        return wordnet.NOUN

def main():
    setup_nltk()

    text = "The students are studying different subjects and they are enjoying their studies. The teachers are teaching the subjects very effectively."
    
    print("=== NLP Basic Operations ===\n")
    print("1. Original Text:")
    print(text)
    print("\n" + "="*50 + "\n")

    # 1. Tokenization
    tokens = word_tokenize(text)
    print("2. Tokens:")
    print(tokens)
    print("\n" + "="*50 + "\n")

    # 2. Stop Words Removal
    # We remove common stop words and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    print("3. Text after stop-word removal:")
    print(filtered_tokens)
    print("\n" + "="*50 + "\n")

    # 3. Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
    print("4. Stemmed words:")
    print(stemmed_words)
    print("\n" + "="*50 + "\n")

    # 4. Lemmatization
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(filtered_tokens)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    print("5. Lemmatized words:")
    print(lemmatized_words)
    print("\n" + "="*50 + "\n")

    # 5. Compare Stemming and Lemmatization
    print("6. Comparison & Answers:")
    print("-" * 30)
    print("Q1: What is the difference between stemming and lemmatization?")
    print("A: Stemming is a crude heuristic process that chops off the ends of words to find the root, often resulting in non-dictionary words (e.g., 'studying' -> 'studi'). Lemmatization uses vocabulary and morphological analysis to properly return the base dictionary form of a word, known as the lemma (e.g., 'studying' -> 'study').\n")
    
    print("Q2: Which technique produces more meaningful English words?")
    print("A: Lemmatization produces more meaningful English words because it maps words to valid dictionary roots.\n")
    
    print("Q3: Why are stop words removed during NLP preprocessing?")
    print("A: Stop words (like 'the', 'are', 'and') occur frequently but don't carry significant semantic meaning. Removing them reduces data size, improves processing speed, and allows algorithms to focus on the important words.\n")
    
    print("Q4: What is the purpose of tokenization?")
    print("A: Tokenization breaks unstructured text into smaller, meaningful units (like words or sentences) called tokens. It is the first step in preparing text for further NLP processing.\n")
    
    print("Q5: Give two real-world applications where these text-processing techniques can be used.")
    print("A: 1. Sentiment Analysis (e.g., analyzing customer feedback or reviews).")
    print("   2. Search Engines / Information Retrieval (e.g., matching user queries to relevant documents).")

if __name__ == "__main__":
    main()
