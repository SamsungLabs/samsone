package com.samsung.samsone.presentation.models

data class ChatMessage(
    val id: String,
    val content: String,
    val isUser: Boolean,
    val timestamp: Long = System.currentTimeMillis(),
    val tokensPerSecond: Float? = null
)

data class AudioFile(
    val name: String,
    val uri: String? = null,
    val resourceId: Int? = null,
    val isFromResources: Boolean = true
)

data class AudioPlayerState(
    val isPlaying: Boolean = false,
    val currentPosition: Long = 0L,
    val duration: Long = 0L,
    val audioFile: AudioFile? = null
)

enum class MessageSource {
    USER,
    MODEL
}
