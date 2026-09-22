plugins {
    kotlin("jvm") version "2.0.21" apply false
    kotlin("plugin.serialization") version "2.0.21" apply false
    kotlin("android") version "2.0.21" apply false
    kotlin("plugin.compose") version "2.0.21" apply false
    id("com.android.application") version "8.7.3" apply false
}

allprojects {
    group = "com.gliksbot.axonhome"
    version = "0.1.0"
}

// Optional local build-dir override for sandboxed environments whose project
// filesystem cannot hold build outputs reliably. On a normal machine this is
// unset and Gradle uses the default per-project build/ directories.
val axonHomeBuildDir: String? = providers.gradleProperty("axonHomeBuildDir").orNull
if (axonHomeBuildDir != null) {
    allprojects {
        @Suppress("DEPRECATION")
        buildDir = File(axonHomeBuildDir, name)
    }
}
