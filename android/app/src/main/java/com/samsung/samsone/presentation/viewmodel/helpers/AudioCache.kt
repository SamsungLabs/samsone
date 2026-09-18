package com.samsung.samsone.presentation.viewmodel.helpers

import org.pytorch.executorch.EValue

/**
 * Manages audio cache state for the audio processing pipeline.
 * Designed to support single or multiple audio files.
 */
class AudioCache {
    private var cacheList: MutableList<AudioProcessingResult> = mutableListOf()
    
    /**
     * Total sequence length across all cached audio files
     */
    var totalSequenceLength: Int = 0
        private set
    
    /**
     * Total embedding dimension across all cached audio files
     */
    var totalEmbeddingDim: Int = 0
        private set
    
    /**
     * Flag indicating if the cache has been prefilled
     */
    var isPrefilled: Boolean = false
        private set
    
    /**
     * Last audio embedding for reuse (from the most recent audio file)
     */
    var lastEmbedding: EValue? = null
        private set
    
    /**
     * Number of audio files currently in cache
     */
    val size: Int
        get() = cacheList.size
    
    /**
     * Checks if the cache is empty
     */
    val isEmpty: Boolean
        get() = cacheList.isEmpty()
    
    /**
     * Gets the most recently cached audio processing result
     */
    val latest: AudioProcessingResult?
        get() = cacheList.lastOrNull()
    
    /**
     * Adds a processed audio result to the cache
     */
    fun add(result: AudioProcessingResult) {
        cacheList.add(result)
        updateTotalDimensions()
    }
    
    /**
     * Clears all cached audio data
     */
    fun clear() {
        cacheList.clear()
        totalSequenceLength = 0
        totalEmbeddingDim = 0
        isPrefilled = false
        lastEmbedding = null
    }
    
    /**
     * Resets the cache state without clearing results
     */
    fun resetState() {
        totalSequenceLength = 0
        totalEmbeddingDim = 0
        isPrefilled = false
        lastEmbedding = null
    }
    
    /**
     * Updates the cache as prefilled with the latest audio data
     */
    fun setPrefilled(result: AudioProcessingResult) {
        if (cacheList.isEmpty()) {
            cacheList.add(result)
        } else {
            cacheList[cacheList.size - 1] = result
        }
        totalSequenceLength = result.sequenceLength
        totalEmbeddingDim = result.embeddingDim
        lastEmbedding = result.lastEmbedding
        isPrefilled = true
    }
    
    /**
     * Gets a specific cached audio result by index
     */
    fun get(index: Int): AudioProcessingResult? {
        return cacheList.getOrNull(index)
    }
    
    /**
     * Updates total dimensions based on cached results
     */
    private fun updateTotalDimensions() {
        totalSequenceLength = cacheList.sumOf { it.sequenceLength }
        totalEmbeddingDim = cacheList.maxOfOrNull { it.embeddingDim } ?: 0
        lastEmbedding = cacheList.lastOrNull()?.lastEmbedding
    }
}
