plugins {
    id("com.android.application")
    kotlin("android")
    kotlin("plugin.compose")
    kotlin("plugin.serialization")
}

// Native Android UI for Axon Home. Runs on the deterministic simulator in
// :core behind the :controlplane client contract; NO network libraries in v1.
android {
    namespace = "com.gliksbot.axonhome"
    compileSdk = 35
    buildToolsVersion = "35.0.0"

    defaultConfig {
        applicationId = "com.gliksbot.axonhome"
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = providers.gradleProperty("axonHomeVersion").getOrElse("0.2.0-local-foundation")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    flavorDimensions += "runtime"
    productFlavors {
        create("local") {
            dimension = "runtime"
            applicationIdSuffix = ".local"
        }
        create("simulation") {
            dimension = "runtime"
            applicationIdSuffix = ".simulation"
            versionNameSuffix = "-simulation"
            resValue("string", "app_name", "Axon Home · SIMULATION")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        debug {
            // Debug builds are never minified; keep everything for inspection.
        }
    }

    testOptions { unitTests.isIncludeAndroidResources = true }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    packaging {
        resources {
            excludes += setOf("META-INF/DEPENDENCIES", "META-INF/*.kotlin_module")
        }
    }
}

// Git commit for the in-app About/version display (arch §10.T).
val gitCommit: String = providers.environmentVariable("GITHUB_SHA").orElse("local").get()

android {
    defaultConfig {
        buildConfigField("String", "GIT_COMMIT", "\"${gitCommit.take(12)}\"")
        buildConfigField("String", "SIM_SEED", "\"20260919\"")
    }
}

dependencies {
    implementation(project(":core"))
    implementation(project(":controlplane"))

    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3:1.3.1")
    implementation("androidx.compose.material3:material3-window-size-class:1.3.1")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.navigation:navigation-compose:2.8.5")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.datastore:datastore-preferences:1.1.1")
    implementation("androidx.biometric:biometric:1.1.0")

    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.6.1")

    debugImplementation("androidx.compose.ui:ui-tooling")

    // JVM unit tests only; no emulator/instrumented tests in this slice
    // (documented in the delivery report: no emulator image is available in
    // the build sandbox, and all state mapping is JVM-testable by design).
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.14.1")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
    testImplementation("app.cash.turbine:turbine:1.1.0")
}
