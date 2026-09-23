package com.samsung.samsone.presentation.viewmodel

import android.annotation.SuppressLint
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import com.samsung.samsone.Tokenizer
import com.samsung.samsone.presentation.models.AudioFile
import com.samsung.samsone.presentation.models.AudioPlayerState
import com.samsung.samsone.presentation.models.ChatMessage
import com.samsung.samsone.presentation.models.SALMConfig
import com.samsung.samsone.presentation.viewmodel.helpers.AudioCache
import com.samsung.samsone.presentation.viewmodel.helpers.AudioProcessingResult
import com.samsung.samsone.presentation.viewmodel.helpers.ProcessingUtils
import com.samsung.samsone.presentation.viewmodel.helpers.PromptUtils
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.IOException
import javax.inject.Inject

@HiltViewModel
class MainScreenViewModel @Inject constructor(
    @ApplicationContext private val context: Context
) : ViewModel() {

    private val _uiState = MutableStateFlow<UiState>(UiState.Loading(""))
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _audioPlayerState = MutableStateFlow(AudioPlayerState())
    val audioPlayerState: StateFlow<AudioPlayerState> = _audioPlayerState.asStateFlow()

    private val _chatMessages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val chatMessages: StateFlow<List<ChatMessage>> = _chatMessages.asStateFlow()

    private val _currentQuery = MutableStateFlow("")
    val currentQuery: StateFlow<String> = _currentQuery.asStateFlow()

    private val _currentConfig = MutableStateFlow(SALMConfig.Default)
    val currentConfig: StateFlow<SALMConfig> = _currentConfig.asStateFlow()
    
    private val _selectedConfigForSwitch = MutableStateFlow(SALMConfig.Default)
    
    private val _pruneVocab = MutableStateFlow(SALMConfig.Default.enablePruning)
    val pruneVocab: StateFlow<Boolean> = _pruneVocab.asStateFlow()

    // Custom model state
    private val _customAudioPath = MutableStateFlow("")
    val customAudioPath: StateFlow<String> = _customAudioPath.asStateFlow()
    
    private val _customTextPath = MutableStateFlow("")
    val customTextPath: StateFlow<String> = _customTextPath.asStateFlow()
    
    private val _customEnablePruning = MutableStateFlow(false)
    val customEnablePruning: StateFlow<Boolean> = _customEnablePruning.asStateFlow()

    internal lateinit var audioLM: Module
    internal lateinit var textModel: Module
    internal lateinit var tokenizer: Tokenizer
    
    internal lateinit var pruneMap: Map<Int, Int>
    internal lateinit var reversePruneMap: Map<Int, Int>

    private var currentAudioFile: AudioFile? = null
    private lateinit var exoPlayer: ExoPlayer
    private lateinit var processedAudio: EValue
    
    // Audio cache management
    private val audioCache = AudioCache()

    init {
        exoPlayer = ExoPlayer.Builder(context).build()
        
        currentAudioFile = AudioFile(
            name = "Water",
            resourceId = com.samsung.samsone.R.raw.water,
            isFromResources = true
        )
        _audioPlayerState.value = _audioPlayerState.value.copy(
            audioFile = currentAudioFile
        )
        
        loadAudioFile(currentAudioFile!!)

        loadModels()
    }

    override fun onCleared() {
        super.onCleared()
        exoPlayer.release()
    }

    private fun loadModels() {
        viewModelScope.launch {
            try {
                withContext(Dispatchers.Main) {
                    _uiState.value = UiState.Loading("Loading tokenizer...")
                }

                initTokenizer()

                // Load models using current config
                loadModelsForConfig(_currentConfig.value)

                processAudioAndPrefillCache(currentAudioFile!!)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    _uiState.value = UiState.Error("Failed to load models: ${e.message}")
                }
            }
        }
    }

    fun updateQuery(query: String) {
        _currentQuery.value = query
    }

    fun togglePruneVocab(enabled: Boolean) {
        _pruneVocab.value = enabled
    }

    fun processUserQuery(query: String) {
        if (query.isBlank()) return
        
        viewModelScope.launch {
            try {
                val startTime = System.currentTimeMillis()
                
                val userMessage = ChatMessage(
                    id = System.currentTimeMillis().toString(),
                    content = query,
                    isUser = true
                )
                
                _chatMessages.value = listOf(userMessage)
                withContext(Dispatchers.Main) {
                    _chatMessages.value = listOf(userMessage)
                    _uiState.value = UiState.Processing("")
                }
                
                withContext(Dispatchers.Default) {
                    var textTokens = tokenizer.encode(PromptUtils.normalize(query) + " answer: ")
                    
                    // Apply reverse prune map if pruning is enabled
                    textTokens = ProcessingUtils.applyReversePruneMap(
                        textTokens,
                        reversePruneMap,
                        _pruneVocab.value
                    )
                    
                    val tokensTensor = ProcessingUtils.createTensorFromTokens(textTokens)
                    val queryEmbeddings = audioLM.execute("embedding", EValue.from(tokensTensor))[0]
                    Log.i("MODEL", "QUERY EMBEDDINGS GENERATED")

                    val textSeqLen = queryEmbeddings.toTensor().shape()[1].toInt()
                    val textRangeArray = ProcessingUtils.createPositionRange(audioCache.totalSequenceLength, textSeqLen)
                    val textInputPosTensor = ProcessingUtils.createTensorFromPositions(textRangeArray)

                    val firstTokenOutput = textModel.forward(EValue.from(textInputPosTensor), queryEmbeddings)
                    Log.i("MODEL", "KV CACHE EXTENDED WITH TEXT EMBEDDINGS: $textSeqLen tokens")

                    val totalSeqLen = audioCache.totalSequenceLength + textSeqLen
                    var currentPosition = totalSeqLen
                    
                    var maxIdx = audioLM.execute("argmax", firstTokenOutput[0])[0].toTensor().dataAsLongArray[0].toInt()
                    
                    Log.i("MODEL", "Starting generation from position $currentPosition with audio cache pre-filled")
                    var tokenCount = 0
                    var decodedTokens = ""
                    
                    val modelMessage = ChatMessage(
                        id = (System.currentTimeMillis() + 1).toString(),
                        content = "",
                        isUser = false
                    )
                    val lastTokenEValue = EValue.from(Tensor.fromBlob(intArrayOf(maxIdx), longArrayOf(1, 1)))
                    var lastEmbedding = audioLM.execute("embedding", lastTokenEValue)[0]
                    
                    if (maxIdx != 0) {
                        textTokens += maxIdx

                        val decodedToken = ProcessingUtils.decodeTokenWithPruneMap(
                            maxIdx,
                            pruneMap,
                            _pruneVocab.value,
                            tokenizer
                        )
                        decodedTokens += decodedToken
                        tokenCount = 1
                        currentPosition++
                        
                        withContext(Dispatchers.Main) {
                            val updatedMessage = modelMessage.copy(content = decodedTokens)
                            _chatMessages.value = listOf(userMessage, updatedMessage)
                            _uiState.value = UiState.Processing(decodedTokens)
                        }
                        
                        Log.i("MODEL", "Generated token ($maxIdx): $decodedToken")
                    }
                    
                    while (maxIdx != 0 && currentPosition <= 500) {
                        val inputPosTensor = Tensor.fromBlob(longArrayOf(currentPosition.toLong()), longArrayOf(1))
                        
                        val output = textModel.forward(EValue.from(inputPosTensor), lastEmbedding)
                        maxIdx = audioLM.execute("argmax", output[0])[0].toTensor().dataAsLongArray[0].toInt()

                        textTokens += maxIdx

                        val lastTokenEValue = EValue.from(Tensor.fromBlob(intArrayOf(maxIdx), longArrayOf(1, 1)))
                        lastEmbedding = audioLM.execute("embedding", lastTokenEValue)[0]

                        if (maxIdx != 0) {
                            val decodedToken = ProcessingUtils.decodeTokenWithPruneMap(
                                maxIdx,
                                pruneMap,
                                _pruneVocab.value,
                                tokenizer
                            )
                            decodedTokens += decodedToken
                        }
                        tokenCount++
                        currentPosition++
                        
                        val elapsedTimeInSeconds = (System.currentTimeMillis() - startTime) / 1000.0f
                        val tokensPerSecond = if (elapsedTimeInSeconds > 0) tokenCount / elapsedTimeInSeconds else 0f
                        
                        withContext(Dispatchers.Main) {
                            val updatedMessage = modelMessage.copy(
                                content = decodedTokens,
                                tokensPerSecond = tokensPerSecond
                            )
                            _chatMessages.value = listOf(userMessage, updatedMessage)
                            _uiState.value = UiState.Processing(decodedTokens)
                        }
                    }
                    
                    withContext(Dispatchers.Main) {
                        val elapsedTimeInSeconds = (System.currentTimeMillis() - startTime) / 1000.0f
                        val finalTokensPerSecond = if (elapsedTimeInSeconds > 0) tokenCount / elapsedTimeInSeconds else 0f
                        val finalMessage = modelMessage.copy(
                            content = decodedTokens,
                            tokensPerSecond = finalTokensPerSecond
                        )
                        _chatMessages.value = listOf(userMessage, finalMessage)
                        _uiState.value = UiState.ModelsReady
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    _uiState.value = UiState.Error("Failed to process query: ${e.message}")
                }
            }
        }
    }

    fun selectAudioFile(audioFile: AudioFile) {
        currentAudioFile = audioFile
        _chatMessages.value = listOf()
        _audioPlayerState.value = _audioPlayerState.value.copy(
            audioFile = audioFile,
            isPlaying = false,
            currentPosition = 0L
        )
        loadAudioFile(audioFile)
        
        // Reset audio cache when new audio is selected
        audioCache.resetState()
        
        // Pre-process audio in background and prefill cache
        viewModelScope.launch {
            processAudioAndPrefillCache(audioFile)
        }
    }

    private fun loadAudioFile(audioFile: AudioFile) {
        val mediaItem = when {
            audioFile.isFromResources && audioFile.resourceId != null -> {
                val resourceUri = "${context.packageName}/raw/${context.resources.getResourceEntryName(audioFile.resourceId)}"
                MediaItem.fromUri(Uri.parse("android.resource://$resourceUri"))
            }
            audioFile.uri != null -> {
                MediaItem.fromUri(Uri.parse(audioFile.uri))
            }
            else -> null
        }
        
        mediaItem?.let {
            exoPlayer.setMediaItem(it)
            exoPlayer.prepare()
        }
    }

    fun getExoPlayer(): ExoPlayer = exoPlayer
    
    /**
     * Selects a model for switching (temporary selection, doesn't load yet)
     */
    fun selectModelForSwitch(config: SALMConfig) {
        _selectedConfigForSwitch.value = config
    }
    
    /**
     * Cancels the model switch and resets to current model
     */
    fun cancelModelSwitch() {
        _selectedConfigForSwitch.value = _currentConfig.value
    }
    
    /**
     * Sets custom audio model path
     */
    fun setCustomAudioPath(path: String) {
        _customAudioPath.value = path
    }
    
    /**
     * Sets custom text model path
     */
    fun setCustomTextPath(path: String) {
        _customTextPath.value = path
    }
    
    /**
     * Sets custom model pruning setting
     */
    fun setCustomEnablePruning(enabled: Boolean) {
        _customEnablePruning.value = enabled
    }
    
    /**
     * Confirms and loads the selected model
     */
    fun confirmModelSwitch() {
        val newConfig = _selectedConfigForSwitch.value
        if (newConfig.name == _currentConfig.value.name) {
            return // Same model, no need to reload
        }
        
        // Validation for Custom model
        if (newConfig.name == "Custom") {
            if (_customAudioPath.value.isEmpty() || _customTextPath.value.isEmpty()) {
                // Treat as cancel - don't load
                Log.w("MODEL", "Custom model selected but paths not provided. Cancelling.")
                cancelModelSwitch()
                return
            }
        }
        
        viewModelScope.launch {
            try {
                withContext(Dispatchers.Main) {
                    _uiState.value = UiState.Loading("Loading ${newConfig.name}...")
                }
                
                // Clear chat history
                _chatMessages.value = emptyList()
                
                // Determine the config to use
                val configToLoad = if (newConfig.name == "Custom") {
                    SALMConfig(
                        name = "Custom",
                        audioModelPath = _customAudioPath.value,
                        textModelPath = _customTextPath.value,
                        enablePruning = _customEnablePruning.value
                    )
                } else {
                    newConfig
                }
                
                // Load models with config
                loadModelsForConfig(configToLoad)
                
                // Update current config (use the actual loaded config)
                _currentConfig.value = configToLoad
                _selectedConfigForSwitch.value = configToLoad
                
                // Update prune vocab based on config
                _pruneVocab.value = configToLoad.enablePruning
                
                // Clear audio cache
                audioCache.resetState()
                
                // Reprocess current audio
                processAudioAndPrefillCache(currentAudioFile!!)
                
                Log.i("MODEL", "Switched to ${configToLoad.name} successfully")
            } catch (e: Exception) {
                Log.e("MODEL", "Failed to switch model: ${e.message}")
                withContext(Dispatchers.Main) {
                    _uiState.value = UiState.Error("Failed to switch model: ${e.message}")
                }
            }
        }
    }
    
    /**
     * Loads models for a specific configuration
     */
    private suspend fun loadModelsForConfig(config: SALMConfig) {
        withContext(Dispatchers.Main) {
            _uiState.value = UiState.Loading("Loading audio model...")
        }
        
        val audioLMPath = config.audioModelPath
        audioLM = withContext(Dispatchers.Default) {
            Module.load(audioLMPath)
        }
        Log.i("MODEL", "Audio model loaded: ${config.name}")
        
        withContext(Dispatchers.Main) {
            _uiState.value = UiState.Loading("Loading text model...")
        }
        
        val textModelPath = config.textModelPath
        textModel = withContext(Dispatchers.Default) {
            Module.load(textModelPath)
        }
        Log.i("MODEL", "Text model loaded: ${config.name}")
        
        // Load prune maps if pruning is enabled
        if (config.enablePruning) {
            withContext(Dispatchers.Main) {
                _uiState.value = UiState.Loading("Loading prune maps...")
            }
            loadPruneMaps()
            Log.i("MODEL", "Prune maps loaded for ${config.name}")
        } else {
            // Clear prune maps if pruning is disabled
            pruneMap = emptyMap()
            reversePruneMap = emptyMap()
            Log.i("MODEL", "Prune maps disabled for ${config.name}")
        }
    }
    
    /**
     * Loads and processes audio file data from an AudioFile (resource or URI)
     */
    private suspend fun loadAndProcessAudio(audioFile: AudioFile): AudioProcessingResult {
        return withContext(Dispatchers.Default) {
            val audio = if (audioFile.isFromResources && audioFile.resourceId != null) {
                loadWav16kMono(context, audioFile.resourceId)
            } else if (audioFile.uri != null) {
                loadWavAndResampleTo16k(context, Uri.parse(audioFile.uri))
            } else {
                // Fallback to default
                loadWav16kMono(context, com.samsung.samsone.R.raw.water)
            }
            val audioTensor = Tensor.fromBlob(audio, longArrayOf(audio.size.toLong()))
            val processed = audioLM.execute("processor", EValue.from(audioTensor))[0]
            AudioProcessingResult(
                processedAudio = processed,
                sequenceLength = 0,
                embeddingDim = 0,
                lastEmbedding = null
            )
        }
    }
    
    /**
     * Gets the prune map for benchmarking
     */
    fun getPruneMap(): Map<Int, Int> = pruneMap
    
    /**
     * Gets the reverse prune map for benchmarking
     */
    fun getReversePruneMap(): Map<Int, Int> = reversePruneMap
    
    /**
     * Gets the enable pruning flag for benchmarking
     */
    fun getEnablePruning(): Boolean = _pruneVocab.value
    
    /**
     * Loads 16kHz mono WAV audio from a URI
     */
    private fun loadWavAndResampleTo16k(context: Context, uri: Uri): FloatArray {
        val inputStream = context.contentResolver.openInputStream(uri)
        val data = inputStream!!.readBytes()
        inputStream.close()

        val sourceRate = (data[27].toInt() and 0xff shl 24) or
                (data[26].toInt() and 0xff shl 16) or
                (data[25].toInt() and 0xff shl 8) or
                (data[24].toInt() and 0xff)

        val channels = ((data[23].toInt() and 0xff) shl 8) or (data[22].toInt() and 0xff)
        val bitsPerSample = ((data[35].toInt() and 0xff) shl 8) or (data[34].toInt() and 0xff)

        require(bitsPerSample == 16) { "Only 16-bit PCM supported" }
        val targetRate = 16000

        val pcmStart = 44
        val sourceSamplesCount = (data.size - pcmStart) / (2 * channels)
        val sourceData = FloatArray(sourceSamplesCount)

        for (i in 0 until sourceSamplesCount) {
            val baseIdx = pcmStart + (i * 2 * channels)
            val lo = data[baseIdx].toInt() and 0xff
            val hi = data[baseIdx + 1].toInt() and 0xff
            val sample = (hi shl 8) or lo
            val signedSample = if (sample >= 32768) sample - 65536 else sample
            sourceData[i] = signedSample / 32768.0f // Normalize to [-1.0, 1.0]
        }

        if (sourceRate == targetRate) return sourceData

        val ratio = sourceRate.toDouble() / targetRate.toDouble()
        val targetSamplesCount = (sourceSamplesCount / ratio).toInt()
        val resampledData = FloatArray(targetSamplesCount)

        for (i in 0 until targetSamplesCount) {
            val sourcePos = i * ratio
            val index = sourcePos.toInt()
            val fraction = (sourcePos - index).toFloat()

            if (index + 1 < sourceData.size) {
                resampledData[i] = (1 - fraction) * sourceData[index] + fraction * sourceData[index + 1]
            } else {
                resampledData[i] = sourceData[index]
            }
        }

        return resampledData
    }
    
    /**
     * Processes audio and pre-fills the cache in a single operation
     */
    private suspend fun processAudioAndPrefillCache(audioFile: AudioFile) {
        withContext(Dispatchers.Main) {
            _uiState.value = UiState.Loading("Processing audio...")
        }
        
        withContext(Dispatchers.Default) {
            try {
                val processingResult = loadAndProcessAudio(audioFile)
                processedAudio = processingResult.processedAudio
                
                val cacheResult = prefillCacheForAudio(processingResult.processedAudio)
                audioCache.setPrefilled(cacheResult)
                
                Log.i("MODEL", "AUDIO PROCESSED AND CACHED: ${cacheResult.sequenceLength} tokens")
                
                withContext(Dispatchers.Main) {
                    _uiState.value = UiState.ModelsReady
                }
            } catch (e: Exception) {
                Log.e("MODEL", "Failed to process audio: ${e.message}")
                withContext(Dispatchers.Main) {
                    _uiState.value = UiState.Error("Failed to process audio: ${e.message}")
                }
            }
        }
    }
    
    /**
     * Pre-fills the KV cache with audio embeddings only
     */
    private suspend fun prefillCacheForAudio(audioToProcess: EValue): AudioProcessingResult {
        return withContext(Dispatchers.Default) {
            val emptyTextTokens = intArrayOf()
            val tokensTensor = ProcessingUtils.createTensorFromTokens(emptyTextTokens)

            Log.i("MODEL", "GENERATING AUDIO-ONLY EMBEDDINGS")
            val audioEmbeddings = audioLM.forward(audioToProcess, EValue.from(tokensTensor))[0]
            Log.i("MODEL", "AUDIO-ONLY EMBEDDINGS GENERATED")

            val seqLen = audioEmbeddings.toTensor().shape()[1].toInt()
            val embeddingDim = audioEmbeddings.toTensor().shape()[2].toInt()

            val positionRange = ProcessingUtils.createPositionRange(0, seqLen)
            val inputPosTensor = ProcessingUtils.createTensorFromPositions(positionRange)

            // Pre-fill KV cache with audio embeddings only
            textModel.forward(EValue.from(inputPosTensor), audioEmbeddings)

            // Extract and store the last audio embedding for reuse
            val lastEmbedding = ProcessingUtils.extractEmbeddingAtIndex(
                audioEmbeddings,
                seqLen - 1,
                seqLen,
                embeddingDim
            )

            Log.i("MODEL", "KV CACHE PREFILLED WITH AUDIO EMBEDDINGS: $seqLen tokens")
            Log.i("MODEL", "LAST AUDIO EMBEDDING STORED FOR REUSE")
            AudioProcessingResult(
                processedAudio = audioToProcess,
                sequenceLength = seqLen,
                embeddingDim = embeddingDim,
                lastEmbedding = lastEmbedding
            )
        }
    }
    
    private fun loadPruneMapFromAsset(assetName: String): Map<Int, Int> {
        val jsonContent = context.assets.open(assetName).bufferedReader().use { it.readText() }
        val jsonObject = JSONObject(jsonContent)
        val map = mutableMapOf<Int, Int>()
        jsonObject.keys().forEach { key ->
            map[key.toInt()] = jsonObject.getInt(key)
        }
        return map
    }

    private fun loadPruneMaps() {
        try {
            pruneMap = loadPruneMapFromAsset("prune_map.json")
            reversePruneMap = loadPruneMapFromAsset("reverse_prune_map.json")

            Log.i("MODEL", "Prune maps loaded successfully")
        } catch (e: Exception) {
            Log.e("MODEL", "Failed to load prune maps: ${e.message}")
            throw e
        }
    }

    suspend fun initTokenizer() {
        withContext(Dispatchers.IO) {
            val tokenizerJsonContent = context.assets.open("tokenizer.json").bufferedReader().use { it.readText() }
            tokenizer = Tokenizer(context)
            tokenizer.init(tokenizerJsonContent)
            Log.i("MODEL", "Tokenizer loaded from assets successfully")
        }
    }

    @Throws(IOException::class)
    private fun getAssetPath(assetName: String): String {
        val modelPath = "/data/local/tmp/$assetName"
        val modelFile = File(modelPath)
        
        if (modelFile.exists()) {
            return modelFile.absolutePath
        } else {
            throw IOException("Model file '$assetName' not found in /data/local/tmp/. Please run 'pushModelsToDevice' gradle task to push the models to the device.")
        }
    }

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

sealed class UiState {
    data class Loading(val message: String) : UiState()
    object ModelsReady : UiState()
    data class Processing(val tokens: String) : UiState()
    data class Error(val message: String) : UiState()
}
