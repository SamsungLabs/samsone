package com.samsung.samsone.presentation.viewmodel

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.samsung.samsone.Tokenizer
import com.samsung.samsone.presentation.models.AudioFile
import com.samsung.samsone.presentation.models.AudioBenchmarkStats
import com.samsung.samsone.presentation.models.BenchmarkConfig
import com.samsung.samsone.presentation.models.BenchmarkMetrics
import com.samsung.samsone.presentation.models.BenchmarkResult
import com.samsung.samsone.presentation.models.BenchmarkState
import com.samsung.samsone.presentation.models.OverallBenchmarkStats
import com.samsung.samsone.presentation.viewmodel.helpers.AudioProcessingResult
import com.samsung.samsone.presentation.viewmodel.helpers.AudioCache
import com.samsung.samsone.presentation.viewmodel.helpers.MemoryTracker
import com.samsung.samsone.presentation.viewmodel.helpers.ProcessingUtils
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import javax.inject.Inject

@HiltViewModel
class BenchmarkViewModel @Inject constructor(
    @ApplicationContext private val context: Context
) : ViewModel() {

    private val _benchmarkState = MutableStateFlow<BenchmarkState>(BenchmarkState.Idle)
    val benchmarkState: StateFlow<BenchmarkState> = _benchmarkState.asStateFlow()

    private val benchmarkConfig = BenchmarkConfig.Default
    private val memoryTracker = MemoryTracker(context)
    
    // Models will be injected/set from MainScreenViewModel
    lateinit var audioLM: Module
    lateinit var textModel: Module
    lateinit var textModelPath: String  // Store path for reloading
    lateinit var tokenizer: Tokenizer
    var pruneMap: Map<Int, Int> = emptyMap()
    var reversePruneMap: Map<Int, Int> = emptyMap()
    var enablePruning: Boolean = false
    
    private val audioCache = AudioCache()
    private val processedAudioCache = mutableMapOf<Int, EValue>()

    /**
     * Runs the complete benchmark suite
     */
    fun runBenchmark() {
        viewModelScope.launch {
            try {
                val totalRuns = benchmarkConfig.audios.size * benchmarkConfig.prompts.size
                val startTime = System.currentTimeMillis()
                
                _benchmarkState.value = BenchmarkState.Running(0, totalRuns)
                
                val allMetrics = mutableListOf<BenchmarkMetrics>()
                var currentRun = 0
                
                // Process each audio file
                for (audio in benchmarkConfig.audios) {
                    val audioFile = AudioFile(
                        name = audio.name,
                        resourceId = audio.resourceId,
                        isFromResources = true
                    )
                    
                    val ramBeforeAudio = getRamUsageMb()
                    val audioPrefillResult = prefillAudioForBenchmark(audioFile)
                    val ramAfterAudioPrefill = getRamUsageMb()
                    
                    for (prompt in benchmarkConfig.prompts) {
                        currentRun++
                        _benchmarkState.value = BenchmarkState.Running(currentRun, totalRuns)
                        
                        // Reload textModel to reset KV cache before each prompt
                        textModel = withContext(Dispatchers.Default) {
                            Module.load(textModelPath)
                        }
                        
                        // Start continuous memory monitoring
                        memoryTracker.startMonitoring(pollingIntervalMs = 100L)
                        
                        // Reprefill audio embeddings after cache reset
                        prefillCacheForAudio(audioPrefillResult.processedAudio)
                        
                        val ramBeforeQuery = getRamUsageMb()
                        
                        val queryPrefillResult = prefillQueryForBenchmark(
                            query = prompt,
                            startSequenceLength = 0,
                            audioSequenceLength = audioPrefillResult.sequenceLength
                        )
                        val ramAfterQueryPrefill = getRamUsageMb()
                        
                        val generationResult = generateTokensForBenchmark(
                            queryPrefillResult = queryPrefillResult,
                            audioSequenceLength = audioPrefillResult.sequenceLength
                        )
                        
                        // Stop memory monitoring and get peak RAM
                        val peakRamDuringInference = memoryTracker.stopMonitoring()
                        
                        val ramAfterGeneration = getRamUsageMb()
                        
                        // Decode tokens after timing is complete
                        val generatedOutput = decodeTokens(generationResult.generatedTokens)
                        
                        // Calculate total duration for this run
                        val totalDurationMs = audioPrefillResult.timeMs + queryPrefillResult.timeMs + generationResult.timeMs
                        
                        // Calculate energy in Wh using average power draw and time
                        val totalDurationHours = totalDurationMs / 1000.0 / 3600.0

                        val metrics = BenchmarkMetrics(
                            audioName = audio.name,
                            prompt = prompt,
                            audioPrefillTimeMs = audioPrefillResult.timeMs,
                            queryPrefillTimeMs = queryPrefillResult.timeMs,
                            generationTimeMs = generationResult.timeMs,
                            tokensGenerated = generationResult.tokensGenerated,
                            generatedTokens = generationResult.generatedTokens,
                            generatedOutput = generatedOutput,
                            ramBeforeMb = ramBeforeQuery,
                            ramAfterAudioPrefillMb = ramAfterAudioPrefill,
                            ramAfterQueryPrefillMb = ramAfterQueryPrefill,
                            ramAfterGenerationMb = ramAfterGeneration,
                            ramPeakDuringGenerationMb = peakRamDuringInference,
                        )
                        
                        allMetrics.add(metrics)
                    }
                    
                    // Clear audio cache for next audio file
                    processedAudioCache.remove(audio.resourceId)
                    
                    // Reload textModel to reset KV cache for next audio file
                    textModel = withContext(Dispatchers.Default) {
                        Module.load(textModelPath)
                    }
                    Log.i("BENCHMARK", "TextModel reloaded to reset KV cache for next audio")
                }
                
                val totalTime = System.currentTimeMillis() - startTime
                val result = calculateBenchmarkResults(allMetrics, totalTime)
                
                _benchmarkState.value = BenchmarkState.Completed(result)
                
                Log.i("BENCHMARK", "Benchmark completed: ${totalRuns} runs in ${totalTime}ms")
                
            } catch (e: Exception) {
                Log.e("BENCHMARK", "Benchmark failed: ${e.message}", e)
                _benchmarkState.value = BenchmarkState.Error("Benchmark failed: ${e.message}")
            }
        }
    }
    
    /**
     * Optimized audio prefill for benchmarking - no UI updates, just timing
     */
    private suspend fun prefillAudioForBenchmark(audioFile: AudioFile): PrefillMetrics {
        return withContext(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()
            
            val audio = loadWav16kMono(context, audioFile.resourceId!!)
            val audioTensor = Tensor.fromBlob(audio, longArrayOf(audio.size.toLong()))
            
            val processStartTime = System.currentTimeMillis()
            val processedAudio = audioLM.execute("processor", EValue.from(audioTensor))[0]
            
            val embedStartTime = System.currentTimeMillis()
            val emptyTextTokens = intArrayOf()
            val tokensTensor = ProcessingUtils.createTensorFromTokens(emptyTextTokens)
            val audioEmbeddings = audioLM.forward(processedAudio, EValue.from(tokensTensor))[0]
            
            val seqLen = audioEmbeddings.toTensor().shape()[1].toInt()
            val embeddingDim = audioEmbeddings.toTensor().shape()[2].toInt()
            
            val positionRange = ProcessingUtils.createPositionRange(0, seqLen)
            val inputPosTensor = ProcessingUtils.createTensorFromPositions(positionRange)
            
            val prefillStartTime = System.currentTimeMillis()
            textModel.forward(EValue.from(inputPosTensor), audioEmbeddings)
            
            val endTime = System.currentTimeMillis()
            
            // Set audio cache state to track total sequence length
            audioCache.setPrefilled(
                AudioProcessingResult(
                    processedAudio = processedAudio,
                    sequenceLength = seqLen,
                    embeddingDim = embeddingDim,
                    lastEmbedding = null
                )
            )
            
            // Cache the processed audio for reuse across prompts
            processedAudioCache[audioFile.resourceId!!] = processedAudio
            
            PrefillMetrics(
                processedAudio = processedAudio,
                sequenceLength = seqLen,
                timeMs = endTime - processStartTime
            )
        }
    }
    
    /**
     * Optimized query prefill for benchmarking
     * Returns the first token output to continue generation properly
     */
    private suspend fun prefillQueryForBenchmark(
        query: String,
        startSequenceLength: Int,
        audioSequenceLength: Int
    ): QueryPrefillMetrics {
        return withContext(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()
            
            var textTokens = tokenizer.encode(query + " answer: ")
            textTokens = ProcessingUtils.applyReversePruneMap(
                textTokens,
                reversePruneMap,
                enablePruning
            )
            
            val tokensTensor = ProcessingUtils.createTensorFromTokens(textTokens)
            val queryEmbeddings = audioLM.execute("embedding", EValue.from(tokensTensor))[0]
            
            val textSeqLen = queryEmbeddings.toTensor().shape()[1].toInt()
            // Start position should be audio sequence length (not startSequenceLength which is 0)
            val textRangeArray = ProcessingUtils.createPositionRange(audioSequenceLength, textSeqLen)
            val textInputPosTensor = ProcessingUtils.createTensorFromPositions(textRangeArray)
            
            val firstTokenOutput = textModel.forward(EValue.from(textInputPosTensor), queryEmbeddings)
            
            val endTime = System.currentTimeMillis()
            
            // Get the first token from the output to continue generation
            val firstTokenId = audioLM.execute("argmax", firstTokenOutput[0])[0].toTensor().dataAsLongArray[0].toInt()
            
            // Create embedding for the first token to continue generation
            val firstTokenEValue = EValue.from(Tensor.fromBlob(intArrayOf(firstTokenId), longArrayOf(1, 1)))
            val firstTokenEmbedding = audioLM.execute("embedding", firstTokenEValue)[0]
            
            QueryPrefillMetrics(
                sequenceLength = textSeqLen,
                timeMs = endTime - startTime,
                firstTokenId = firstTokenId,
                firstTokenEmbedding = firstTokenEmbedding
            )
        }
    }
    
    /**
     * Optimized token generation for benchmarking
     * Tracks generated tokens list, times generation, then decodes output after timing
     * Starts from the actual last token from query prefill (not dummy token)
     */
    private suspend fun generateTokensForBenchmark(
        queryPrefillResult: QueryPrefillMetrics,
        audioSequenceLength: Int,
        maxTokens: Int = 300
    ): GenerationMetrics {
        return withContext(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()
            
            // Start from audio sequence length + query sequence length
            val startPosition = audioSequenceLength + queryPrefillResult.sequenceLength
            var currentPosition = startPosition
            var maxIdx = queryPrefillResult.firstTokenId
            var tokenCount = 0
            
            // Track all generated tokens
            val generatedTokens = mutableListOf<Int>()
            
            // Start from the actual last embedding from query prefill
            var lastEmbedding = queryPrefillResult.firstTokenEmbedding
            
            // Track peak RAM during generation
            var peakRamMb = getRamUsageMb()
            
            // Generate maxTokens tokens
            // Start with the first token we got from query prefill
            if (maxIdx != 0) {
                generatedTokens.add(maxIdx)
                tokenCount = 1
                currentPosition++
            }
            
            while (tokenCount < maxTokens && currentPosition < 500) {
                val inputPosTensor = Tensor.fromBlob(longArrayOf(currentPosition.toLong()), longArrayOf(1))
                
                val output = textModel.forward(EValue.from(inputPosTensor), lastEmbedding)
                maxIdx = audioLM.execute("argmax", output[0])[0].toTensor().dataAsLongArray[0].toInt()
                
                if (maxIdx == 0) break
                
                generatedTokens.add(maxIdx)
                
                val lastTokenEValue = EValue.from(Tensor.fromBlob(intArrayOf(maxIdx), longArrayOf(1, 1)))
                lastEmbedding = audioLM.execute("embedding", lastTokenEValue)[0]
                
                tokenCount++
                currentPosition++
                
                // Track peak RAM
                peakRamMb = maxOf(peakRamMb, getRamUsageMb())
            }
            
            val endTime = System.currentTimeMillis()
            
            GenerationMetrics(
                tokensGenerated = tokenCount,
                generatedTokens = generatedTokens.toList(),
                timeMs = endTime - startTime,
                peakRamMb = peakRamMb
            )
        }
    }
    
    /**
     * Decodes a list of token IDs using the tokenizer
     */
    private fun decodeTokens(tokens: List<Int>): String {
        return tokens.mapNotNull { token ->
            ProcessingUtils.decodeTokenWithPruneMap(
                token,
                pruneMap,
                enablePruning,
                tokenizer
            )
        }.joinToString("")
    }
    
    /**
     * Prefills cache with processed audio (helper function)
     */
    private suspend fun prefillCacheForAudio(processedAudio: EValue) {
        withContext(Dispatchers.Default) {
            val emptyTextTokens = intArrayOf()
            val tokensTensor = ProcessingUtils.createTensorFromTokens(emptyTextTokens)
            val audioEmbeddings = audioLM.forward(processedAudio, EValue.from(tokensTensor))[0]
            
            val seqLen = audioEmbeddings.toTensor().shape()[1].toInt()
            val embeddingDim = audioEmbeddings.toTensor().shape()[2].toInt()
            
            val positionRange = ProcessingUtils.createPositionRange(0, seqLen)
            val inputPosTensor = ProcessingUtils.createTensorFromPositions(positionRange)
            
            textModel.forward(EValue.from(inputPosTensor), audioEmbeddings)
            
            audioCache.setPrefilled(
                AudioProcessingResult(
                    processedAudio = processedAudio,
                    sequenceLength = seqLen,
                    embeddingDim = embeddingDim,
                    lastEmbedding = null
                )
            )
        }
    }
    
    /**
     * Calculates overall and per-audio statistics from all metrics
     */
    private fun calculateBenchmarkResults(
        allMetrics: List<BenchmarkMetrics>,
        totalTimeMs: Long
    ): BenchmarkResult {
        // Calculate overall averages
        val avgAudioPrefill = allMetrics.map { it.audioPrefillTimeMs }.average()
        val avgQueryPrefill = allMetrics.map { it.queryPrefillTimeMs }.average()
        val avgTokensPerSec = allMetrics.map { it.tokensPerSecond }.average()
        val avgTokensGenerated = allMetrics.map { it.tokensGenerated }.average()
        val maxRam = allMetrics.maxOfOrNull { it.ramPeakDuringGenerationMb } ?: 0L

        val overallStats = OverallBenchmarkStats(
            avgAudioPrefillTimeMs = avgAudioPrefill,
            avgQueryPrefillTimeMs = avgQueryPrefill,
            avgTokensPerSecond = avgTokensPerSec,
            avgTokensGenerated = avgTokensGenerated,
            maxRamUsageMb = maxRam,
            totalRuns = allMetrics.size,
            totalTimeMs = totalTimeMs
        )
        
        // Group by audio and calculate per-audio stats
        val perAudioStats = allMetrics.groupBy { it.audioName }.mapValues { (audioName, metrics) ->
            AudioBenchmarkStats(
                audioName = audioName,
                avgAudioPrefillTimeMs = metrics.map { it.audioPrefillTimeMs }.average(),
                avgQueryPrefillTimeMs = metrics.map { it.queryPrefillTimeMs }.average(),
                avgTokensPerSecond = metrics.map { it.tokensPerSecond }.average(),
                avgTokensGenerated = metrics.map { it.tokensGenerated }.average(),
                runCount = metrics.size
            )
        }
        
        return BenchmarkResult(
            overallAverages = overallStats,
            perAudioStats = perAudioStats,
            individualRuns = allMetrics
        )
    }
    
    /**
     * Gets current RAM usage in MB using ActivityManager.MemoryInfo
     * This captures system-wide memory including native library allocations
     */
    private fun getRamUsageMb(): Long {
        return memoryTracker.getCurrentMemoryUsageMb()
    }
    
    /**
     * Loads 16kHz mono WAV audio from resources
     */
    private fun loadWav16kMono(context: Context, resId: Int): FloatArray {
        val inputStream = context.resources.openRawResource(resId)
        val data = inputStream.readBytes()
        inputStream.close()

        // WAV header parsing
        val sampleRate =
            (data[27].toInt() shl 24) or
                    ((data[26].toInt() and 0xff) shl 16) or
                    ((data[25].toInt() and 0xff) shl 8) or
                    (data[24].toInt() and 0xff)

        val channels =
            ((data[23].toInt() and 0xff) shl 8) or
                    (data[22].toInt() and 0xff)

        val bitsPerSample =
            ((data[35].toInt() and 0xff) shl 8) or
                    (data[34].toInt() and 0xff)

        require(sampleRate == 16000) { "Expected 16kHz, got $sampleRate" }
        require(channels == 1) { "Expected mono, got $channels" }
        require(bitsPerSample == 16) { "Expected 16-bit PCM" }

        val pcmStart = 44
        val samples = (data.size - pcmStart) / 2
        val audio = FloatArray(samples)

        var idx = pcmStart
        for (i in 0 until samples) {
            val lo = data[idx].toInt() and 0xff
            val hi = data[idx + 1].toInt() and 0xff
            val sample = (hi shl 8) or lo
            val signedSample =
                if (sample >= 32768) sample - 65536 else sample
            audio[i] = signedSample / 32768.0f
            idx += 2
        }
        return audio
    }
}

/**
 * Result of audio prefill
 */
private data class PrefillMetrics(
    val processedAudio: EValue,
    val sequenceLength: Int,
    val timeMs: Long
)

/**
 * Result of query prefill - now includes first token for proper generation continuation
 */
private data class QueryPrefillMetrics(
    val sequenceLength: Int,
    val timeMs: Long,
    val firstTokenId: Int,
    val firstTokenEmbedding: EValue
)

/**
 * Result of token generation - now includes peak RAM and generated tokens
 */
private data class GenerationMetrics(
    val tokensGenerated: Int,
    val generatedTokens: List<Int>,
    val timeMs: Long,
    val peakRamMb: Long
)
