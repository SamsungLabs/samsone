package com.samsung.samsone.presentation.models

/**
 * Metrics for a single audio + prompt benchmark run
 */
data class BenchmarkMetrics(
    val audioName: String,
    val prompt: String,
    val audioPrefillTimeMs: Long,
    val queryPrefillTimeMs: Long,
    val generationTimeMs: Long,
    val tokensGenerated: Int,
    val generatedTokens: List<Int>,
    val generatedOutput: String,
    val ramBeforeMb: Long,
    val ramAfterAudioPrefillMb: Long,
    val ramAfterQueryPrefillMb: Long,
    val ramAfterGenerationMb: Long,
    val ramPeakDuringGenerationMb: Long,
) {
    val tokensPerSecond: Double
        get() = if (generationTimeMs > 0) {
            (tokensGenerated.toDouble() / generationTimeMs.toDouble()) * 1000.0
        } else {
            0.0
        }
    
    val ramDeltaAudioPrefillMb: Long
        get() = ramAfterAudioPrefillMb - ramBeforeMb
    
    val ramDeltaQueryPrefillMb: Long
        get() = ramAfterQueryPrefillMb - ramAfterAudioPrefillMb
    
    val ramDeltaGenerationMb: Long
        get() = ramAfterGenerationMb - ramAfterQueryPrefillMb
}

/**
 * Statistics for a single audio file across all prompts
 */
data class AudioBenchmarkStats(
    val audioName: String,
    val avgAudioPrefillTimeMs: Double,
    val avgQueryPrefillTimeMs: Double,
    val avgTokensPerSecond: Double,
    val avgTokensGenerated: Double,
    val runCount: Int
)

/**
 * Overall benchmark statistics across all audios and prompts
 */
data class OverallBenchmarkStats(
    val avgAudioPrefillTimeMs: Double,
    val avgQueryPrefillTimeMs: Double,
    val avgTokensPerSecond: Double,
    val avgTokensGenerated: Double,
    val maxRamUsageMb: Long,
    val totalRuns: Int,
    val totalTimeMs: Long
)

/**
 * Complete benchmark results
 */
data class BenchmarkResult(
    val overallAverages: OverallBenchmarkStats,
    val perAudioStats: Map<String, AudioBenchmarkStats>,
    val individualRuns: List<BenchmarkMetrics>
)

/**
 * State for benchmark execution
 */
sealed class BenchmarkState {
    object Idle : BenchmarkState()
    data class Running(val currentRun: Int, val totalRuns: Int) : BenchmarkState()
    data class Completed(val result: BenchmarkResult) : BenchmarkState()
    data class Error(val message: String) : BenchmarkState()
}
