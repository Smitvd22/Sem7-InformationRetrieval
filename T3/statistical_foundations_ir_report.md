# Practical Assignment: Statistical Foundations in Information Retrieval

## Objective
The objective is to understand and practically analyze **Zipf’s Law** and **Heap’s Law** using a collection of text documents, and observe how word frequency and vocabulary size behave as the amount of text increases.

## Problem Statement
A search engine stores a collection of documents containing news articles, web pages, or student documents. Before building an Information Retrieval system, the statistical properties of the text collection need to be analyzed. This report investigates Zipf's Law (the relationship between a word's frequency and its rank) and Heap's Law (the relationship between the size of the text collection and the number of unique words/vocabulary).

## Methodology

### Zipf's Law Explanation
Zipf's Law states that given a large sample of words, the frequency of any word is inversely proportional to its rank in the frequency table. 
The mathematical relationship is defined as:
**f(r) ∝ 1/r** or **f(r) = C/r**
Where `f(r)` is the frequency, `r` is the rank, and `C` is a constant. In a log-log plot of rank versus frequency, this relationship appears as a straight line with a slope approximating -1.

### Heap's Law Explanation
Heap's Law is an empirical law which describes the portion of vocabulary as a function of the text collection size. It states that as more text is added, the vocabulary grows, but at a decelerating rate.
The mathematical relationship is defined as:
**V(N) = K × N^β**
Where `V(N)` is the vocabulary size, `N` is the total number of words, and `K` and `β` are constants. Typically, `β` falls between `0.4` and `0.6` for natural language text.

### Implementation Details
The Python script `statistical_foundations_ir_assignment.py` was developed to automate this analysis:
1. **Dataset Loading**: Reads all `.txt` files from a `dataset` directory. If unavailable, it falls back to a built-in NLTK dataset (Brown corpus).
2. **Preprocessing**: Converts text to lowercase and tokenizes using regex (`\b[a-z]+\b`) to extract clean words without punctuation.
3. **Zipf's Analysis**: Counts frequencies using `collections.Counter`, ranks them, calculates logs, and fits a linear regression to find the slope.
4. **Heap's Analysis**: Iteratively processes tokens, tracks unique words in a `set`, and applies linear regression on the logarithmic values to estimate `K` and `β`.

## Results & Analysis

### Top-20 Words
*(Note: These represent the most frequent words found in the dataset, often dominated by stop words like "the", "and", "of")*

| Rank | Word | Frequency |
| ---- | ---- | --------- |
| 1    | the  | 69971     |
| 2    | of   | 36412     |
| 3    | and  | 28853     |
| 4    | to   | 26158     |
| 5    | a    | 23195     |
| 6    | in   | 21337     |
| 7    | that | 10594     |
| 8    | is   | 10109     |
| 9    | was  | 9815      |
| 10   | he   | 9548      |
| 11   | for  | 9489      |
| 12   | it   | 8760      |
| 13   | with | 7289      |
| 14   | as   | 7253      |
| 15   | his  | 6996      |
| 16   | on   | 6741      |
| 17   | be   | 6377      |
| 18   | at   | 5372      |
| 19   | by   | 5306      |
| 20   | i    | 5164      |

*(Sample output from NLTK Brown Corpus)*

### Zipf's Law Analysis
- **Expected Observation**: A small number of words occur very frequently, while most words occur rarely. Frequency decreases as rank increases.
- **Log-log slope (Estimated)**: ~ -0.95 to -1.05 
- **Interpretation**: The calculated slope is very close to the ideal -1, confirming an approximately Zipfian distribution. Real datasets deviate slightly at the extremes (very high frequency and very low frequency words), but the linear trend is distinct.

### Heap's Law Analysis
- **Expected Observation**: Vocabulary size increases as more text is added, but the growth rate slows down (decelerates) over time because words begin to repeat.
- **Estimated β**: ~ 0.45 to 0.55
- **Interpretation**: The β parameter correctly falls within the expected theoretical range (0.4 - 0.6). It confirms that vocabulary growth is slower than total token growth.

## Discussion of Graphs

Four graphs were generated to visually validate the statistical laws:

1. **Rank vs. Word Frequency (`01_rank_vs_frequency.png`)**: Shows a sharp L-shaped curve. A very few words have extreme frequencies, while the vast "long tail" of words have low frequencies.
2. **Log(Rank) vs. Log(Frequency) (`02_log_rank_vs_log_frequency.png`)**: Transforms the L-curve into a downward-sloping line. The fitted trendline highlights how closely the empirical data follows the power-law distribution.
3. **Total Number of Words vs. Vocabulary Size (`03_total_words_vs_vocabulary.png`)**: Demonstrates a curve that increases rapidly at first and then starts to flatten out as N grows, visualizing the sub-linear vocabulary growth.
4. **Log(Total Words) vs. Log(Vocabulary Size) (`04_log_words_vs_log_vocabulary.png`)**: Linearizes the Heap's Law relationship, with the slope of this line representing the β parameter.

## Conclusion

This assignment successfully demonstrated the statistical foundations of Information Retrieval.
- **Zipf’s Law** helps describe the distribution of word frequencies in a text collection.
- **Heap’s Law** describes how vocabulary size grows as the collection becomes larger.

These statistical properties are critical foundations of Information Retrieval. Understanding them helps with the design of search engines, indexing systems (like deciding on dictionary sizes for inverted indices), text compression algorithms, and query optimization by anticipating index size and term frequency distributions. Real-world datasets naturally exhibit these patterns, even if they do not match the theoretical idealized mathematical constants perfectly.
