# BM25: A Probabilistic Relevance Framework for Information Retrieval

**Based on:** Robertson and Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond"

## Summary

BM25 (Best Matching 25) is a ranking function used by search engines to estimate the relevance of documents to a given search query. It is based on the probabilistic retrieval framework developed in the 1970s and 1980s. Despite being decades old, BM25 remains one of the most effective and widely used ranking functions in information retrieval, often serving as a strong baseline that modern neural approaches must beat.

## How BM25 Works

BM25 scores a document D with respect to a query Q containing terms q1, q2, ..., qn. The score is computed as the sum of the contributions of each query term:

For each query term qi, the score contribution depends on three factors:

1. **Term Frequency (TF):** How often the query term appears in the document. BM25 uses a saturating TF function — the benefit of additional occurrences diminishes. This is controlled by the parameter k1 (typically 1.2 to 2.0). Higher k1 means TF saturation happens more slowly.

2. **Inverse Document Frequency (IDF):** Terms that appear in fewer documents are more informative. BM25 uses a logarithmic IDF formula that gives higher weight to rare terms. A term appearing in every document contributes nothing to the score.

3. **Document Length Normalization:** Longer documents naturally contain more term occurrences. The parameter b (between 0 and 1, typically 0.75) controls how much document length affects the score. When b=1, full length normalization is applied. When b=0, no length normalization occurs.

## Key Parameters

- **k1 (term frequency saturation):** Controls how quickly the contribution of term frequency saturates. Default is typically 1.2. Lower values mean faster saturation (diminishing returns from repeated terms).
- **b (length normalization):** Controls document length normalization. Default is 0.75. A value of 0 ignores document length entirely; a value of 1 fully scales by document length.
- **k3 (query term frequency):** Relevant when query terms can repeat. In most practical applications this is set to 0 or ignored since queries are short.

## Strengths of BM25

1. **No training required:** BM25 is unsupervised and requires no labeled data. It works immediately on any text corpus with no fine-tuning.
2. **Exact match:** BM25 excels at finding documents containing the exact query terms. This is complementary to semantic search which finds conceptually similar but lexically different content.
3. **Interpretability:** The scoring is transparent — you can see exactly why a document ranked high based on which terms matched and their frequencies.
4. **Efficiency:** BM25 scoring with inverted indices is extremely fast, even on large corpora with millions of documents.

## Limitations

1. **Vocabulary mismatch:** BM25 cannot match semantically related terms. If a document says "automobile" and the query says "car," BM25 will not match them.
2. **No semantic understanding:** Word order and meaning are ignored. "Dog bites man" and "man bites dog" score identically for a query about dog bites.
3. **Tokenization sensitivity:** Results depend heavily on how text is tokenized. Inconsistent tokenization between documents and queries is a common source of bugs in retrieval systems.

## BM25 in Hybrid Search

Modern retrieval systems often combine BM25 with dense retrieval (neural embedding similarity) in a hybrid approach. This leverages the complementary strengths of both methods: BM25 handles exact keyword matching while dense retrieval handles semantic similarity. Fusion methods like Reciprocal Rank Fusion (RRF) merge the ranked lists from both approaches into a single ranking.
