plugins {
    kotlin("jvm") version "2.0.21"
    kotlin("plugin.serialization") version "2.0.21"
}

group = "com.gliksbot.axonhome"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.6.1")

    testImplementation(kotlin("test"))
    testImplementation("junit:junit:4.13.2")
}

kotlin {
    jvmToolchain(17)
}

tasks.test {
    useJUnitPlatform()
    // The contract tests read the repo-level controlplane/ tree (examples/ and
    // schemas/). Resolution order in the tests: JVM system property
    // CONTROLPLANE_ROOT -> env var -> walk up from user.dir. We set the system
    // property here; override with -PcontrolplaneRoot=<path> for standalone
    // builds where the repo-level tree lives elsewhere.
    val controlplaneRoot = providers.gradleProperty("controlplaneRoot")
        .orElse(provider { File(projectDir, "../../..").canonicalPath })
    systemProperty("CONTROLPLANE_ROOT", controlplaneRoot.get())
}
