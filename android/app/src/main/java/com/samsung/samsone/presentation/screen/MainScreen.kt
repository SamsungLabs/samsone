package com.samsung.samsone.presentation.screen

import android.content.Context
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.activity.result.contract.ActivityResultContracts
import com.samsung.samsone.presentation.components.AudioFilePicker
import com.samsung.samsone.presentation.components.AudioPlayer
import com.samsung.samsone.presentation.components.ChatMessageList
import com.samsung.samsone.presentation.components.ModelSelectorDialog
import com.samsung.samsone.presentation.models.SALMConfig
import com.samsung.samsone.presentation.viewmodel.MainScreenViewModel
import com.samsung.samsone.presentation.viewmodel.UiState

@Composable
fun MainScreen(
    modifier: Modifier = Modifier,
    viewModel: MainScreenViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    val audioPlayerState by viewModel.audioPlayerState.collectAsState()
    val chatMessages by viewModel.chatMessages.collectAsState()
    val currentQuery by viewModel.currentQuery.collectAsState()
    val pruneVocab by viewModel.pruneVocab.collectAsState()
    val currentConfig by viewModel.currentConfig.collectAsState()
    val customAudioPath by viewModel.customAudioPath.collectAsState()
    val customTextPath by viewModel.customTextPath.collectAsState()
    val customEnablePruning by viewModel.customEnablePruning.collectAsState()
    
    val context = LocalContext.current
    
    var showAudioPicker by remember { mutableStateOf(false) }
    var showModelSelector by remember { mutableStateOf(false) }
    var showBenchmarkScreen by remember { mutableStateOf(false) }
    var userQuery by remember { mutableStateOf("") }
    var selectedConfig by remember { mutableStateOf(currentConfig) }
    
    val audioFileLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val path = getPathFromUri(context, it)
            if (path.endsWith(".pte", ignoreCase = true)) {
                viewModel.setCustomAudioPath(path)
            }
        }
    }
    
    // File picker for text model
    val textFileLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val path = getPathFromUri(context, it)
            if (path.endsWith(".pte", ignoreCase = true)) {
                viewModel.setCustomTextPath(path)
            }
        }
    }

    LaunchedEffect(currentQuery) {
        userQuery = currentQuery
    }
    
    // Reset selected config when dialog is dismissed without confirmation
    LaunchedEffect(showModelSelector) {
        if (!showModelSelector) {
            selectedConfig = currentConfig
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(8.dp)
    ) {
        if (showBenchmarkScreen) {
            BenchmarkScreen(
                onBackClick = { showBenchmarkScreen = false },
                modelName = currentConfig.name
            )
            return@Column
        }

        when (val state = uiState) {
            is UiState.Loading -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        CircularProgressIndicator()
                        Text(
                            text = state.message,
                            style = MaterialTheme.typography.bodyLarge
                        )
                    }
                }
            }

            is UiState.ModelsReady, is UiState.Processing -> {
                // Main app layout
                Column(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    AudioPlayer(
                        audioPlayerState = audioPlayerState,
                        player = viewModel.getExoPlayer()
                    )

                    // Model name display
                    Text(
                        text = currentConfig.name,
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.align(Alignment.CenterHorizontally)
                    )

                    ChatMessageList(
                        messages = chatMessages,
                        isProcessing = state is UiState.Processing,
                        modifier = Modifier.weight(1f)
                    )

                    if (showAudioPicker) {
                        AudioFilePicker(
                            currentAudioFile = audioPlayerState.audioFile,
                            onAudioFileSelected = { audioFile ->
                                viewModel.selectAudioFile(audioFile)
                                showAudioPicker = false
                            }
                        )
                    }

                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical=16.dp),
                            verticalAlignment = Alignment.Bottom,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            IconButton(
                                onClick = { showAudioPicker = !showAudioPicker },
                                modifier = Modifier.size(48.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Add,
                                    contentDescription = "Add Audio"
                                )
                            }

                            IconButton(
                                onClick = { showBenchmarkScreen = true },
                                modifier = Modifier.size(48.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Analytics,
                                    contentDescription = "Benchmark"
                                )
                            }

                            IconButton(
                                onClick = { showModelSelector = true },
                                modifier = Modifier.size(48.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Settings,
                                    contentDescription = "Model Settings"
                                )
                            }

                            OutlinedTextField(
                                value = userQuery,
                                onValueChange = {
                                    userQuery = it
                                    viewModel.updateQuery(it)
                                },
                                label = { Text("Enter your query") },
                                modifier = Modifier.weight(1f),
                                enabled = state !is UiState.Processing,
                                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                                keyboardActions = KeyboardActions(
                                    onSend = {
                                        if (userQuery.isNotBlank()) {
                                            val uQ = userQuery
                                            userQuery = ""
                                            viewModel.processUserQuery(uQ)
                                        }
                                    }
                                )
                            )

                            IconButton(
                                onClick = {
                                    if (userQuery.isNotBlank()) {
                                        val uQ = userQuery
                                        userQuery = ""
                                        viewModel.processUserQuery(uQ)
                                    }
                                },
                                enabled = userQuery.isNotBlank() && state !is UiState.Processing,
                                modifier = Modifier.size(48.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Send,
                                    contentDescription = "Send"
                                )
                            }
                        }
                    }
                }
            }

            is UiState.Error -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            text = "Error",
                            style = MaterialTheme.typography.headlineMedium,
                            color = MaterialTheme.colorScheme.error
                        )
                        Text(
                            text = state.message,
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }
            }
        }
    }
    
    if (showModelSelector) {
        ModelSelectorDialog(
            currentConfig = currentConfig,
            selectedConfig = selectedConfig,
            configs = SALMConfig.AllConfigs,
            onDismiss = {
                showModelSelector = false
                viewModel.cancelModelSwitch()
            },
            onConfirm = {
                viewModel.confirmModelSwitch()
            },
            onSelectionChanged = { config ->
                selectedConfig = config
                viewModel.selectModelForSwitch(config)
            },
            isCustomSelected = selectedConfig.name == "Custom",
            customAudioPath = customAudioPath,
            customTextPath = customTextPath,
            customEnablePruning = customEnablePruning,
            onAudioBrowse = { audioFileLauncher.launch("*/*") },
            onTextBrowse = { textFileLauncher.launch("*/*") },
            onPruningToggled = { viewModel.setCustomEnablePruning(it) }
        )
    }
}

private fun getPathFromUri(context: Context, uri: Uri): String {
    if (uri.scheme == "file") {
        return uri.path ?: ""
    }
    
    val cursor = context.contentResolver.query(uri, null, null, null, null)
    cursor?.use {
        if (it.moveToFirst()) {
            val index = it.getColumnIndex("_data")
            if (index != -1) {
                return it.getString(index)
            }
        }
    }
    
    return uri.toString()
}
