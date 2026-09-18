import org.gradle.kotlin.dsl.support.serviceOf

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.hilt.android)
    id("kotlin-kapt")
}


tasks.register("pushModelsToDevice") {
    group = "custom"
    description = "Pushes AudioLLM's .pte files directly to the app's internal files directory."

    val exportPath = "${project.rootDir}/../outputs/export/"
    val audioModel99MFileName = "99M/exported/Samsone99M_audio_model.pte"
    val textModel99MFileName = "99M/exported/Samsone99M_text_model.pte"
    val audioModel134MFileName = "134M/exported/Samsone134M_audio_model.pte"
    val textModel134MFileName = "134M/exported/Samsone134M_text_model.pte"
    val audioModel356MFileName = "356M/exported/Samsone356M_audio_model.pte"
    val textModel356MFileName = "356M/exported/Samsone356M_text_model.pte"
    val aM99MPath = "${exportPath}${audioModel99MFileName}"
    val tM99MPath = "${exportPath}${textModel99MFileName}"
    val aM134MPath = "${exportPath}${audioModel134MFileName}"
    val tM134MPath = "${exportPath}${textModel134MFileName}"
    val aM356MPath = "${exportPath}${audioModel356MFileName}"
    val tM356MPath = "${exportPath}${textModel356MFileName}"

    val devicePath = "/data/local/tmp/"

    val execOps = project.serviceOf<ExecOperations>()

    doLast {
        val models = listOf(
            aM99MPath,
            tM99MPath,
            aM134MPath,
            tM134MPath,
            aM356MPath,
            tM356MPath,
        )

        models.forEach { modelPath ->
            execOps.exec {
                commandLine("adb", "push", modelPath, devicePath)
            }
        }
    }

}

android {
    namespace = "com.samsung.samsone"
    compileSdk = 35


    defaultConfig {
        applicationId = "com.samsung.samsone"
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags("-std=c++17")
                abiFilters("arm64-v8a", "armeabi-v7a", "x86_64")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.executorch.android)
    implementation(libs.squareup.okhttp)
    implementation(libs.hilt.android)
    implementation(libs.hilt.navigation.compose)
    implementation(libs.androidx.media3.exoplayer)
    implementation(libs.androidx.media3.exoplayer.dash)
    implementation(libs.androidx.media3.ui)
    implementation(libs.androidx.media3.ui.compose)
    implementation(libs.androidx.media3.ui.compose.material3)
    implementation(libs.androidx.material.icons.extended)
    kapt(libs.hilt.android.compiler)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}
