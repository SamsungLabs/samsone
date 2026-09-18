package com.samsung.samsone.presentation.models

import com.samsung.samsone.R

data class BenchmarkConfig(
    val audios: List<BenchmarkAudio>,
    val prompts: List<String>
) {
    companion object {
        val Default = BenchmarkConfig(
            audios = listOf(
                BenchmarkAudio(name = "Water", resourceId = R.raw.water),
                BenchmarkAudio(name = "Train", resourceId = R.raw.train),
                BenchmarkAudio(name = "Cow", resourceId = R.raw.cow)
            ),
            prompts = listOf(
                "caption the audio",
                "describe the content of the audio",
                "in which environment the audio was recorded?",
                "what is the main theme of the audio?",
                "is there anything specific in the audio?"
            )
        )
    }
}

data class BenchmarkAudio(
    val name: String,
    val resourceId: Int
)
