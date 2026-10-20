# Attention Is All You Need

**Authors:** Vaswani et al. (2017)
**Paper:** https://arxiv.org/abs/1706.03762

## Summary

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. This paper proposes the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

## Key Contributions

1. **Self-Attention Mechanism:** The Transformer uses multi-head self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. Each attention head learns to attend to information from different representation subspaces at different positions.

2. **Positional Encoding:** Since the model contains no recurrence and no convolution, positional encodings are added to the input embeddings to give the model information about the relative or absolute position of the tokens in the sequence. The authors use sine and cosine functions of different frequencies.

3. **Scaled Dot-Product Attention:** The attention function maps a query and a set of key-value pairs to an output. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. Scaling by the square root of the dimension prevents the dot products from growing large in magnitude.

4. **Multi-Head Attention:** Instead of performing a single attention function, the model linearly projects the queries, keys and values h times with different learned linear projections. These are then concatenated and projected again to produce the final values.

## Architecture Details

The Transformer follows an encoder-decoder structure. The encoder maps an input sequence of symbol representations to a sequence of continuous representations. The decoder then generates an output sequence of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input.

The encoder consists of a stack of 6 identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism, and a position-wise fully connected feed-forward network. Residual connections are employed around each sub-layer, followed by layer normalization.

The decoder also consists of a stack of 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer which performs multi-head attention over the output of the encoder stack. The self-attention sub-layer in the decoder is modified to prevent positions from attending to subsequent positions (masking).

## Results

The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over existing best results by over 2 BLEU. On the WMT 2014 English-to-French translation task, the model achieves a new single-model state-of-the-art BLEU score of 41.0. Training took 3.5 days on 8 P100 GPUs.

## Impact

The Transformer architecture became the foundation for nearly all modern large language models including BERT, GPT, T5, and their successors. The self-attention mechanism proved to be more parallelizable than recurrent layers and required significantly less time to train.
