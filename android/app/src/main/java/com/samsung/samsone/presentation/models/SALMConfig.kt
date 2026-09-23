package com.samsung.samsone.presentation.models

/**
 * Configuration class for app Audio Language Models (SALM)
 * 
 * @property name Display name of the model (e.g., "Samsone99M", "Samsone134M")
 * @property audioModelPath Path to the audio model file
 * @property textModelPath Path to the text model file
 * @property enablePruning Whether vocabulary pruning should be enabled for this model
 */
data class SALMConfig(
    val name: String,
    val audioModelPath: String,
    val textModelPath: String,
    val enablePruning: Boolean
) {
    companion object {
        val Samsone99M = SALMConfig(
            name = "Samsone99M",
            audioModelPath = "/data/local/tmp/Samsone99M_audio_model.pte",
            textModelPath = "/data/local/tmp/Samsone99M_text_model.pte",
            enablePruning = true
        )

        val Samsone134M = SALMConfig(
            name = "Samsone134M",
            audioModelPath = "/data/local/tmp/Samsone134M_audio_model.pte",
            textModelPath = "/data/local/tmp/Samsone134M_text_model.pte",
            enablePruning = true
        )

        val Samsone356M = SALMConfig(
            name = "Samsone356M",
            audioModelPath = "/data/local/tmp/Samsone356M_audio_model.pte",
            textModelPath = "/data/local/tmp/Samsone356M_text_model.pte",
            enablePruning = true
        )

        val Custom = SALMConfig(
            name = "Custom",
            audioModelPath = "",
            textModelPath = "",
            enablePruning = false
        )

        val Default = Samsone134M

        val AllConfigs = listOf(Samsone99M, Samsone134M, Samsone356M)
    }
}
