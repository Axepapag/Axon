pluginManagement {
    // maven.google.com may be unreachable in some sandboxes; the identical
    // content is served at https://dl.google.com/dl/android/maven2. CI uses
    // the default (maven.google.com); local builds export GOOGLE_MAVEN_URL.
    val googleMavenUrl: String =
        providers.environmentVariable("GOOGLE_MAVEN_URL").orElse("https://maven.google.com").get()
    repositories {
        maven { url = uri(googleMavenUrl) }
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    val googleMavenUrl: String =
        providers.environmentVariable("GOOGLE_MAVEN_URL").orElse("https://maven.google.com").get()
    repositories {
        maven { url = uri(googleMavenUrl) }
        mavenCentral()
    }
}

rootProject.name = "axon-home"

// :core is the pure-Kotlin JVM heart of Axon Home, :controlplane the pure
// Kotlin Control Plane client contract, :app the native Android UI.
include(":core", ":controlplane", ":app")
