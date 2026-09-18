package com.samsung.samsone

import android.content.Context

class Tokenizer(private val context: Context) {
    private var nativeHandle: Long = 0

    companion object {
        init {
            System.loadLibrary("native-lib")
        }
    }

    fun init(json: String) {
        nativeHandle = nativeInit(json)
    }

    fun encode(text: String): IntArray = nativeEncode(nativeHandle, text)

    fun decode(ids: IntArray): String = nativeDecode(nativeHandle, ids)

    fun release() {
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0
        }
    }

    private external fun nativeInit(json: String): Long
    private external fun nativeEncode(handle: Long, text: String): IntArray
    private external fun nativeDecode(handle: Long, ids: IntArray): String
    private external fun nativeDestroy(handle: Long)
}