package com.samsung.samsone.presentation.viewmodel.helpers

import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor

/**
 * Utility functions for common processing operations in the audio pipeline.
 */
object ProcessingUtils {
    
    /**
     * Applies reverse prune map to token array if pruning is enabled.
     * @param tokens Original token array
     * @param reversePruneMap Map for reverse pruning
     * @param pruneEnabled Whether pruning is enabled
     * @return Transformed token array
     */
    fun applyReversePruneMap(
        tokens: IntArray,
        reversePruneMap: Map<Int, Int>,
        pruneEnabled: Boolean
    ): IntArray {
        return if (pruneEnabled) {
            tokens.map { token ->
                reversePruneMap[token] ?: token
            }.toIntArray()
        } else {
            tokens
        }
    }
    
    /**
     * Decodes a single token with optional prune map application.
     * @param token Token to decode
     * @param pruneMap Map for pruning
     * @param pruneEnabled Whether pruning is enabled
     * @param tokenizer Tokenizer instance for decoding
     * @return Decoded token string
     */
    fun decodeTokenWithPruneMap(
        token: Int,
        pruneMap: Map<Int, Int>,
        pruneEnabled: Boolean,
        tokenizer: com.samsung.samsone.Tokenizer
    ): String {
        val tokenToDecode = if (pruneEnabled) {
            pruneMap[token] ?: token
        } else {
            token
        }
        return tokenizer.decode(intArrayOf(tokenToDecode))
    }
    
    /**
     * Creates a tensor from a token array with batch dimension.
     * @param tokens Token array
     * @return Tensor with shape [1, tokens.size]
     */
    fun createTensorFromTokens(tokens: IntArray): Tensor {
        return Tensor.fromBlob(tokens, longArrayOf(1, tokens.size.toLong()))
    }
    
    /**
     * Creates a tensor from a position array.
     * @param positions Position array
     * @return Tensor with shape [positions.size]
     */
    fun createTensorFromPositions(positions: LongArray): Tensor {
        return Tensor.fromBlob(positions, longArrayOf(positions.size.toLong()))
    }
    
    /**
     * Creates a position tensor for a range starting from a given value.
     * @param start Starting position
     * @param length Number of positions
     * @return LongArray with positions
     */
    fun createPositionRange(start: Int, length: Int): LongArray {
        return LongArray(length) { (start + it).toLong() }
    }
    
    /**
     * Extracts an embedding at a specific index from batch embeddings.
     * @param batchEmbeddings Batch embeddings EValue
     * @param index Index of the embedding to extract
     * @param sequenceLength Total sequence length
     * @param embeddingDim Embedding dimension
     * @return EValue containing the single embedding
     */
    fun extractEmbeddingAtIndex(
        batchEmbeddings: EValue,
        index: Int,
        sequenceLength: Int,
        embeddingDim: Int
    ): EValue {
        val embeddingsData = batchEmbeddings.toTensor().dataAsFloatArray
        val singleEmbeddingData = FloatArray(embeddingDim)
        System.arraycopy(embeddingsData, index * embeddingDim, singleEmbeddingData, 0, embeddingDim.toInt())
        return EValue.from(Tensor.fromBlob(singleEmbeddingData, longArrayOf(1, 1, embeddingDim.toLong())))
    }
}
