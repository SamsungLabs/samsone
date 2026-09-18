package com.samsung.samsone.presentation.components

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Agriculture
import androidx.compose.material.icons.filled.AirplanemodeActive
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material.icons.filled.Flight
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Train
import androidx.compose.material.icons.filled.WaterDrop
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.samsung.samsone.presentation.models.AudioFile

@Composable
fun AudioFilePicker(
    currentAudioFile: AudioFile?,
    onAudioFileSelected: (AudioFile) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    
    // Launcher for picking audio files from device
    val audioPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { 
            val fileName = getFileNameFromUri(context, it) ?: "Selected Audio"
            onAudioFileSelected(
                AudioFile(
                    name = fileName,
                    uri = it.toString(),
                    isFromResources = false
                )
            )
        }
    }

    Card(
        modifier = modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(4.dp)
        ) {
            Text(
                text = "Audio Source",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(bottom = 12.dp)
            )
            
            // Current audio file info
            currentAudioFile?.let { audioFile ->
                Text(
                    text = "Current: ${audioFile.name}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            }
            
            Divider(modifier = Modifier.padding(vertical = 8.dp))
            
            // Audio selection buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                IconButton(
                    onClick = {
                        onAudioFileSelected(
                            AudioFile(
                                name = "Water",
                                resourceId = com.samsung.samsone.R.raw.water,
                                isFromResources = true
                            )
                        )
                    },
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.WaterDrop,
                        contentDescription = null,
                        modifier = Modifier.width(16.dp)
                    )
                }
                IconButton(
                    onClick = {
                        onAudioFileSelected(
                            AudioFile(
                                name = "Plane",
                                resourceId = com.samsung.samsone.R.raw.airport,
                                isFromResources = true
                            )
                        )
                    },
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.AirplanemodeActive,
                        contentDescription = null,
                        modifier = Modifier.width(16.dp)
                    )
                }

                IconButton(
                    onClick = {
                        onAudioFileSelected(
                            AudioFile(
                                name = "Cow",
                                resourceId = com.samsung.samsone.R.raw.cow,
                                isFromResources = true
                            )
                        )
                    },
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.Agriculture,
                        contentDescription = null,
                        modifier = Modifier.width(16.dp)
                    )
                }
                
                OutlinedButton(
                    onClick = {
                        audioPickerLauncher.launch("audio/*")
                    },
                    modifier = Modifier.weight(3f)
                ) {
                    Icon(
                        imageVector = Icons.Default.Folder,
                        contentDescription = null,
                        modifier = Modifier.width(16.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Device Files")
                }
            }
        }
    }
}

private fun getFileNameFromUri(context: android.content.Context, uri: android.net.Uri): String? {
    return try {
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            cursor.moveToFirst()
            cursor.getString(nameIndex)
        }
    } catch (e: Exception) {
        null
    }
}
