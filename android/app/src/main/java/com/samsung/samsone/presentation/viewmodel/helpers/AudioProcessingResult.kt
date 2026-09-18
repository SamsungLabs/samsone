package com.samsung.samsone.presentation.viewmodel.helpers

import org.pytorch.executorch.EValue

data class AudioProcessingResult(
    val processedAudio: EValue,
    val sequenceLength: Int,
    val embeddingDim: Int,
    val lastEmbedding: EValue?
) {
    companion object {
        fun empty() = AudioProcessingResult(
            processedAudio = EValue.from(0L),
            sequenceLength = 0,
            embeddingDim = 0,
            lastEmbedding = null
        )
    }
}
