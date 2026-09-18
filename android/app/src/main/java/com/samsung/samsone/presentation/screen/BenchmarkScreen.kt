package com.samsung.samsone.presentation.screen

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.Divider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.clickable
import androidx.compose.material3.AlertDialog
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.hilt.navigation.compose.hiltViewModel
import com.samsung.samsone.presentation.models.BenchmarkResult
import com.samsung.samsone.presentation.models.BenchmarkState
import com.samsung.samsone.presentation.viewmodel.BenchmarkViewModel
import com.samsung.samsone.presentation.viewmodel.MainScreenViewModel

@Composable
fun BenchmarkScreen(
    onBackClick: () -> Unit,
    modelName: String,
    mainViewModel: MainScreenViewModel = hiltViewModel(),
    viewModel: BenchmarkViewModel = hiltViewModel()
) {
    val benchmarkState by viewModel.benchmarkState.collectAsState()
    
    LaunchedEffect(Unit) {
        viewModel.audioLM = mainViewModel.audioLM
        viewModel.textModel = mainViewModel.textModel
        viewModel.textModelPath = mainViewModel.currentConfig.value.textModelPath
        viewModel.tokenizer = mainViewModel.tokenizer
        viewModel.pruneMap = mainViewModel.getPruneMap()
        viewModel.reversePruneMap = mainViewModel.getReversePruneMap()
        viewModel.enablePruning = mainViewModel.getEnablePruning()
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(8.dp)
    ) {
        // Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            IconButton(onClick = onBackClick) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back"
                )
            }
            Text(
                text = "Benchmark",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold
            )
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Model info
        Text(
            text = "Model: $modelName",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.primary
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        when (val state = benchmarkState) {
            is BenchmarkState.Idle -> {
                BenchmarkIdleContent(onStartBenchmark = { viewModel.runBenchmark() })
            }
            
            is BenchmarkState.Running -> {
                BenchmarkRunningContent(
                    currentRun = state.currentRun,
                    totalRuns = state.totalRuns
                )
            }
            
            is BenchmarkState.Completed -> {
                BenchmarkCompletedContent(
                    result = state.result,
                    onRunAgain = { viewModel.runBenchmark() }
                )
            }
            
            is BenchmarkState.Error -> {
                BenchmarkErrorContent(
                    message = state.message,
                    onRetry = { viewModel.runBenchmark() }
                )
            }
        }
    }
}

@Composable
fun BenchmarkIdleContent(onStartBenchmark: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Ready to run benchmark",
            style = MaterialTheme.typography.titleLarge
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Text(
            text = "The benchmark will run 15 tests:\n" +
                    "• 3 audio files\n" +
                    "• 5 prompts per audio\n" +
                    "\n" +
                    "This will measure:\n" +
                    "• Audio prefill time\n" +
                    "• Query prefill time\n" +
                    "• Tokens per second\n" +
                    "• RAM usage",
            style = MaterialTheme.typography.bodyLarge
        )
        
        Spacer(modifier = Modifier.height(32.dp))
        
        Button(
            onClick = onStartBenchmark,
            modifier = Modifier.size(width = 200.dp, height = 56.dp)
        ) {
            Text("Start Benchmark")
        }
    }
}

@Composable
fun BenchmarkRunningContent(currentRun: Int, totalRuns: Int) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        CircularProgressIndicator(modifier = Modifier.size(64.dp))
        
        Spacer(modifier = Modifier.height(24.dp))
        
        Text(
            text = "Running benchmark...",
            style = MaterialTheme.typography.titleLarge
        )
        
        Spacer(modifier = Modifier.height(8.dp))
        
        Text(
            text = "Test $currentRun of $totalRuns",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
fun BenchmarkCompletedContent(
    result: BenchmarkResult,
    onRunAgain: () -> Unit
) {
    var selectedAudioName by remember { mutableStateOf<String?>(null) }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
    ) {
        // Overall statistics card
        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Overall Results",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                StatRow(
                    label = "Total Runs",
                    value = "${result.overallAverages.totalRuns}"
                )
                
                StatRow(
                    label = "Total Time",
                    value = "${result.overallAverages.totalTimeMs}ms"
                )
                
                StatRow(
                    label = "Avg Audio Prefill",
                    value = "${result.overallAverages.avgAudioPrefillTimeMs.toInt()}ms"
                )
                
                StatRow(
                    label = "Avg Query Prefill",
                    value = "${result.overallAverages.avgQueryPrefillTimeMs.toInt()}ms"
                )
                
                StatRow(
                    label = "Avg Tokens/sec",
                    value = String.format("%.2f", result.overallAverages.avgTokensPerSecond)
                )
                
                StatRow(
                    label = "Avg Tokens Generated",
                    value = String.format("%.1f", result.overallAverages.avgTokensGenerated)
                )
                
                StatRow(
                    label = "Max RAM Usage",
                    value = "${result.overallAverages.maxRamUsageMb}MB"
                )
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Per-audio statistics
        result.perAudioStats.values.forEach { audioStats ->
            AudioStatsCard(
                audioStats = audioStats,
                onClick = { selectedAudioName = audioStats.audioName }
            )
            Spacer(modifier = Modifier.height(16.dp))
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Run again button
        Button(
            onClick = onRunAgain,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Run Again")
        }
        
        // Show outputs dialog when an audio is selected
        selectedAudioName?.let { audioName ->
            val audioMetrics = result.individualRuns.filter { it.audioName == audioName }
            AudioOutputsDialog(
                audioName = audioName,
                metrics = audioMetrics,
                onDismiss = { selectedAudioName = null }
            )
        }
    }
}

@Composable
fun AudioStatsCard(
    audioStats: com.samsung.samsone.presentation.models.AudioBenchmarkStats,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(8.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = audioStats.audioName,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            StatRow(
                label = "Runs",
                value = "${audioStats.runCount}"
            )
            
            StatRow(
                label = "Avg Audio Prefill",
                value = "${audioStats.avgAudioPrefillTimeMs.toInt()}ms"
            )
            
            StatRow(
                label = "Avg Query Prefill",
                value = "${audioStats.avgQueryPrefillTimeMs.toInt()}ms"
            )
            
            StatRow(
                label = "Avg Tokens/sec",
                value = String.format("%.2f", audioStats.avgTokensPerSecond)
            )
            
            StatRow(
                label = "Avg Tokens Generated",
                value = String.format("%.1f", audioStats.avgTokensGenerated)
            )
        }
    }
}

@Composable
fun StatRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun BenchmarkErrorContent(message: String, onRetry: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Benchmark Failed",
            style = MaterialTheme.typography.titleLarge,
            color = MaterialTheme.colorScheme.error
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Text(
            text = message,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.error
        )
        
        Spacer(modifier = Modifier.height(32.dp))
        
        Button(onClick = onRetry) {
            Text("Retry")
        }
    }
}

@Composable
fun AudioOutputsDialog(
    audioName: String,
    metrics: List<com.samsung.samsone.presentation.models.BenchmarkMetrics>,
    onDismiss: () -> Unit
) {
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .padding(16.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier
                    .verticalScroll(rememberScrollState())
            ) {
                // Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = audioName,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    
                    IconButton(onClick = onDismiss) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Close"
                        )
                    }
                }
                
                Divider(modifier = Modifier.padding(vertical = 8.dp))
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Show outputs for each prompt
                metrics.forEach { metric ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp)
                        ) {
                            Text(
                                text = metric.prompt,
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Medium,
                                color = MaterialTheme.colorScheme.primary
                            )
                            
                            Spacer(modifier = Modifier.height(8.dp))
                            
                            // Stats
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = "Tokens: ${metric.tokensGenerated}",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = "${metric.tokensPerSecond.toInt()} tok/s",
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                            
                            Spacer(modifier = Modifier.height(8.dp))
                            
                            Text(
                                text = "Generated Output:",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            
                            Spacer(modifier = Modifier.height(4.dp))
                            
                            Text(
                                text = metric.generatedOutput,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
}
