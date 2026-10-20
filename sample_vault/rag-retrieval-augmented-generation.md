# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Authors:** Lewis et al. (2020)
**Paper:** https://arxiv.org/abs/2005.11401

## Summary

Large pre-trained language models store factual knowledge in their parameters and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is limited, and their performance lags behind task-specific architectures on knowledge-intensive tasks. This paper explores a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models that combine pre-trained parametric and non-parametric memory for language generation.

## Key Contributions

1. **RAG Architecture:** The model combines a pre-trained seq2seq model (the generator) with a dense retrieval component (the retriever) based on Dense Passage Retrieval (DPR). The retriever provides latent documents conditioned on the input, and the seq2seq model conditions on these latent documents together with the input to generate the output.

2. **Two RAG Formulations:** RAG-Sequence uses the same retrieved document to generate the complete output sequence. RAG-Token can use different documents for each output token, allowing the generator to draw information from multiple documents when producing a single answer.

3. **End-to-End Training:** Both the retriever and generator components are jointly trained without any direct supervision on what document should be retrieved. The retriever learns to find documents that help the generator produce correct outputs through backpropagation of the generation loss.

## How It Works

Given an input sequence x, the retriever component p(z|x) returns a distribution over text documents z. The retriever uses a bi-encoder architecture where both the query and document are encoded using BERT-based encoders. The top-k documents are retrieved using Maximum Inner Product Search (MIPS) with a FAISS index.

The generator p(y|x, z) produces the output sequence y given both the original input x and a retrieved document z. The generator is initialized from BART-large, a pre-trained seq2seq transformer.

For RAG-Sequence, the probability of the full output sequence is marginalized over the top-k retrieved documents. For RAG-Token, the marginalization happens at each output token position independently.

## Results

RAG models set new state-of-the-art on three open-domain question answering tasks: Natural Questions, TriviaQA, and WebQuestions. The model also performs well on knowledge-intensive generation tasks like Jeopardy question generation and fact verification with FEVER.

Notably, RAG models generate more specific, diverse, and factual text compared to state-of-the-art parametric-only seq2seq models. On the MSMARCO NLG task, RAG models produce responses that human evaluators prefer over extractive models.

## Why RAG Matters

RAG addresses fundamental limitations of parametric-only language models: their knowledge is fixed at training time, they cannot easily update their knowledge, and they may hallucinate facts. By combining retrieval with generation, RAG models can access up-to-date information, provide attributable answers, and reduce hallucination by grounding generation in retrieved evidence.
