package com.samsung.samsone.presentation.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Divider
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.samsung.samsone.presentation.models.SALMConfig

/**
 * Dialog for selecting a SALM model
 * 
 * @param currentConfig The currently loaded model configuration
 * @param selectedConfig The currently selected model in the dialog (for UI display)
 * @param configs List of available model configurations
 * @param onDismiss Callback when dialog is dismissed/cancelled
 * @param onConfirm Callback when user confirms model selection
 * @param onSelectionChanged Callback when user selects a different model in the list
 * @param isCustomSelected Whether Custom model is currently selected
 * @param customAudioPath Path to custom audio model
 * @param customTextPath Path to custom text model
 * @param customEnablePruning Whether pruning is enabled for custom model
 * @param onAudioBrowse Callback when user clicks browse for audio model
 * @param onTextBrowse Callback when user clicks browse for text model
 * @param onPruningToggled Callback when pruning switch is toggled
 */
@Composable
fun ModelSelectorDialog(
    currentConfig: SALMConfig,
    selectedConfig: SALMConfig,
    configs: List<SALMConfig>,
    onDismiss: () -> Unit,
    onConfirm: () -> Unit,
    onSelectionChanged: (SALMConfig) -> Unit,
    isCustomSelected: Boolean = false,
    customAudioPath: String = "",
    customTextPath: String = "",
    customEnablePruning: Boolean = false,
    onAudioBrowse: () -> Unit = {},
    onTextBrowse: () -> Unit = {},
    onPruningToggled: (Boolean) -> Unit = {}
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(text = "Select Model")
        },
        text = {
            LazyColumn(
                modifier = Modifier.fillMaxWidth()
            ) {
                items(configs.take(2)) { config ->
                    ModelOption(
                        name = config.name,
                        isSelected = config.name == selectedConfig.name,
                        onClick = { onSelectionChanged(config) }
                    )
                }
                
                // Add divider between default models and custom
                item {
                    Divider(
                        modifier = Modifier
                            .padding(vertical = 8.dp)
                            .height(1.dp)
                    )
                }
                
                // Custom model option
                items(configs.takeLast(1)) { config ->
                    Column {
                        ModelOption(
                            name = config.name,
                            isSelected = config.name == selectedConfig.name,
                            onClick = { onSelectionChanged(config) }
                        )
                        
                        // Custom model controls
                        if (isCustomSelected) {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(start = 48.dp, top = 8.dp, bottom = 8.dp)
                            ) {
                                // Audio model path
                                Column(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 4.dp)
                                ) {
                                    Button(
                                        onClick = onAudioBrowse,
                                        shape = RoundedCornerShape(8.dp),
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(40.dp)
                                    ) {
                                        Text("Browse Audio Model")
                                    }
                                    if (customAudioPath.isNotEmpty()) {
                                        Text(
                                            text = "Audio: ${customAudioPath.takeLast(30)}",
                                            style = MaterialTheme.typography.bodySmall,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis,
                                            modifier = Modifier.padding(top = 4.dp)
                                        )
                                    }
                                }
                                
                                Spacer(modifier = Modifier.height(8.dp))
                                
                                // Text model path
                                Column(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 4.dp)
                                ) {
                                    Button(
                                        onClick = onTextBrowse,
                                        shape = RoundedCornerShape(8.dp),
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(40.dp)
                                    ) {
                                        Text("Browse Text Model")
                                    }
                                    if (customTextPath.isNotEmpty()) {
                                        Text(
                                            text = "Text: ${customTextPath.takeLast(30)}",
                                            style = MaterialTheme.typography.bodySmall,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis,
                                            modifier = Modifier.padding(top = 4.dp)
                                        )
                                    }
                                }
                                
                                Spacer(modifier = Modifier.height(8.dp))
                                
                                // Pruning switch
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = "Pruning",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Spacer(modifier = Modifier.width(16.dp))
                                    Switch(
                                        checked = customEnablePruning,
                                        onCheckedChange = onPruningToggled
                                    )
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    onConfirm()
                    onDismiss()
                }
            ) {
                Text("Confirm")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

/**
 * Individual model option with radio button
 */
@Composable
private fun ModelOption(
    name: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 12.dp, horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        RadioButton(
            selected = isSelected,
            onClick = onClick
        )
        Text(
            text = name,
            modifier = Modifier.padding(start = 16.dp)
        )
    }
}
