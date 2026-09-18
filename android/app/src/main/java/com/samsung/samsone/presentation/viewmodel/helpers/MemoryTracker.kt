package com.samsung.samsone.presentation.viewmodel.helpers

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Helper class to track memory usage during benchmarking.
 * Uses ActivityManager.MemoryInfo to capture system-wide memory including native allocations.
 */
class MemoryTracker(private val context: Context) {
    
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
    private var monitoringJob: Job? = null
    private var peakMemoryMb: Long = 0L
    private var isMonitoring = false
    
    /**
     * Gets the current memory usage in MB using ActivityManager.MemoryInfo.
     * This captures system-wide memory including native library allocations.
     * @return Used memory in MB
     */
    fun getCurrentMemoryUsageMb(): Long {
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager?.getMemoryInfo(memoryInfo) ?: return 0L
        
        val totalMem = memoryInfo.totalMem / (1024 * 1024)
        val availableMem = memoryInfo.availMem / (1024 * 1024)
        val usedMem = totalMem - availableMem
        
        return usedMem
    }
    
    /**
     * Gets the current memory usage in MB as a formatted string.
     * @return Memory usage string (e.g., "1024MB")
     */
    fun getMemoryUsageString(): String {
        val usedMem = getCurrentMemoryUsageMb()
        return "${usedMem}MB"
    }
    
    /**
     * Starts continuous memory monitoring in a coroutine.
     * Polls memory usage at the specified interval and tracks the peak.
     * @param pollingIntervalMs Polling interval in milliseconds (default: 100ms)
     */
    fun startMonitoring(pollingIntervalMs: Long = 100L) {
        if (isMonitoring) {
            Log.w("MEMORY_TRACKER", "Memory monitoring is already active")
            return
        }
        
        isMonitoring = true
        peakMemoryMb = 0L
        Log.i("MEMORY_TRACKER", "Starting memory monitoring with ${pollingIntervalMs}ms interval")
        
        // Initial reading
        peakMemoryMb = getCurrentMemoryUsageMb()
        
        // Launch monitoring in a separate coroutine
        monitoringJob = CoroutineScope(Dispatchers.Default).launch {
            while (isActive && isMonitoring) {
                val currentMemory = getCurrentMemoryUsageMb()
                if (currentMemory > peakMemoryMb) {
                    peakMemoryMb = currentMemory
                    Log.d("MEMORY_TRACKER", "New peak memory: ${peakMemoryMb}MB")
                }
                delay(pollingIntervalMs)
            }
        }
    }
    
    /**
     * Stops memory monitoring and returns the peak memory observed.
     * @return Peak memory usage in MB during the monitoring period
     */
    fun stopMonitoring(): Long {
        if (!isMonitoring) {
            Log.w("MEMORY_TRACKER", "Memory monitoring is not active")
            return peakMemoryMb
        }
        
        isMonitoring = false
        monitoringJob?.cancel()
        monitoringJob = null
        
        Log.i("MEMORY_TRACKER", "Stopped memory monitoring. Peak: ${peakMemoryMb}MB")
        return peakMemoryMb
    }
    
    /**
     * Gets the current peak memory without stopping monitoring.
     * @return Current peak memory in MB
     */
    fun getPeakMemory(): Long {
        return peakMemoryMb
    }
    
    /**
     * Resets the peak memory counter.
     */
    fun resetPeak() {
        peakMemoryMb = 0L
        Log.d("MEMORY_TRACKER", "Reset peak memory counter")
    }
    
    /**
     * Checks if monitoring is currently active.
     * @return true if monitoring is active, false otherwise
     */
    fun isMonitoring(): Boolean {
        return isMonitoring
    }
}
