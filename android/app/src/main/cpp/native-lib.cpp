#include <jni.h>
#include <string>
#include <vector>
#include "tokenizers-cpp/include/tokenizers_cpp.h"

using tokenizers::Tokenizer;

// --- Helper: Convert jintArray to std::vector<int> ---
std::vector<int> jintArrayToVector(JNIEnv* env, jintArray array) {
    jsize len = env->GetArrayLength(array);
    std::vector<int> result(len);
    env->GetIntArrayRegion(array, 0, len, result.data());
    return result;
}

// --- Helper: Convert std::vector<int> to jintArray ---
jintArray vectorToJintArray(JNIEnv* env, const std::vector<int>& vec) {
    jintArray result = env->NewIntArray(vec.size());
    env->SetIntArrayRegion(result, 0, vec.size(), vec.data());
    return result;
}

extern "C" {

/**
 * Creates a Tokenizer instance from a JSON string.
 * Returns a long (the memory address of the C++ object).
 */
JNIEXPORT jlong JNICALL
Java_com_app_samsone_Tokenizer_nativeInit(JNIEnv* env, jobject thiz, jstring json_string) {
    const char* c_json = env->GetStringUTFChars(json_string, nullptr);

    //  uses HuggingFace-style JSON.
    // Use FromBlobJSON to load directly from the string content.
    auto tokenizer = Tokenizer::FromBlobJSON(c_json);

    env->ReleaseStringUTFChars(json_string, c_json);

    // We return the raw pointer to Java to keep the object alive in memory.
    return reinterpret_cast<jlong>(tokenizer.release());
}

/**
 * Encodes text into a list of token IDs.
 */
JNIEXPORT jintArray JNICALL
Java_com_app_samsone_Tokenizer_nativeEncode(JNIEnv* env, jobject thiz, jlong handle, jstring text) {
    auto* tokenizer = reinterpret_cast<Tokenizer*>(handle);
    const char* c_text = env->GetStringUTFChars(text, nullptr);

    // Perform the BPE encoding
    std::vector<int> ids = tokenizer->Encode(c_text);

    env->ReleaseStringUTFChars(text, c_text);
    return vectorToJintArray(env, ids);
}

/**
 * Decodes a list of token IDs back into a string.
 */
JNIEXPORT jstring JNICALL
Java_com_app_samsone_Tokenizer_nativeDecode(JNIEnv* env, jobject thiz, jlong handle, jintArray ids) {
    auto* tokenizer = reinterpret_cast<Tokenizer*>(handle);
    std::vector<int> c_ids = jintArrayToVector(env, ids);

    // Perform decoding
    std::string text = tokenizer->Decode(c_ids);

    return env->NewStringUTF(text.c_str());
}

/**
 * Frees the memory when the Tokenizer is no longer needed.
 */
JNIEXPORT void JNICALL
Java_com_app_samsone_Tokenizer_nativeDestroy(JNIEnv* env, jobject thiz, jlong handle) {
delete reinterpret_cast<Tokenizer*>(handle);
}

} // extern "C"