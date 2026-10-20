# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

**Authors:** Reimers and Gurevych (2019)
**Paper:** https://arxiv.org/abs/1908.10084

## Summary

BERT and RoBERTa set new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity. However, both models require that both sentences be fed into the network simultaneously, which causes a massive computational overhead. Finding the most similar pair in a collection of 10,000 sentences requires about 50 million inference computations with BERT, taking roughly 65 hours. Sentence-BERT (SBERT) modifies the pre-trained BERT network using siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine similarity.

## Key Contributions

1. **Siamese Network Architecture:** SBERT adds a pooling operation to the output of BERT to derive a fixed-sized sentence embedding. Three pooling strategies are explored: using the CLS token output, computing the mean of all output vectors (MEAN strategy), and computing the max-over-time of the output vectors. The MEAN strategy consistently achieves the best results.

2. **Training Objectives:** For classification tasks, the model concatenates sentence embeddings u and v with their element-wise difference |u-v| and feeds them through a softmax classifier. For regression tasks (like STS), cosine similarity is computed between the two sentence embeddings and mean squared error loss is used.

3. **Efficient Similarity Search:** Once sentences are embedded, cosine similarity or Euclidean distance can be used to find semantically similar sentences. Reducing the search for the most similar sentence pair from 65 hours (with BERT cross-encoder) to about 5 seconds with SBERT embeddings plus cosine similarity, a speedup of roughly 47,000x.

## Architecture Details

The siamese network ties the weights between two identical BERT models. Both sentences are passed through their respective BERT model independently, producing two sentence embeddings. The key insight is that by training on NLI (Natural Language Inference) data first and then fine-tuning on STS (Semantic Textual Similarity) data, the model learns high-quality sentence representations.

The authors experiment with different pre-trained models (BERT-base, BERT-large, RoBERTa) and find that RoBERTa-based models slightly outperform BERT-based models.

## Results

SBERT achieves competitive performance on STS benchmarks while being orders of magnitude faster for inference. On STS benchmark tasks, SBERT with MEAN pooling achieves a Spearman correlation of 84.67 compared to 84.30 for the BERT cross-encoder approach, while being dramatically faster.

For transfer learning tasks, SBERT embeddings outperform other sentence embedding methods like InferSent and Universal Sentence Encoder on most SentEval transfer tasks.

## Practical Impact

SBERT and its successor models (available through the sentence-transformers library) became the standard approach for generating sentence embeddings in production systems. The all-MiniLM-L6-v2 model, a distilled and optimized variant, offers an excellent balance of speed and quality for semantic search, clustering, and retrieval applications. It produces 384-dimensional embeddings and supports a maximum sequence length of 256 tokens.
