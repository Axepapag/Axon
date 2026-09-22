package com.gliksbot.axonhome.controlplane

import java.io.File

/**
 * Locates the language-neutral contract tree (`controlplane/` at repo level).
 *
 * Resolution order (explicit, in this order):
 * 1. JVM system property `CONTROLPLANE_ROOT` — may point at the repo-level
 *    `controlplane/` dir itself or at its parent (repo root).
 * 2. Environment variable `CONTROLPLANE_ROOT`, same convention.
 * 3. Walk upwards from `user.dir` (the Gradle module working directory,
 *    `android/controlplane/`) looking for a sibling `controlplane/examples`
 *    directory. From the module dir that is `../../../controlplane/`.
 *
 * The test task sets `CONTROLPLANE_ROOT` to the absolute path in
 * build.gradle.kts, so (3) is only a fallback for IDE runs.
 */
object ControlPlanePaths {

    fun root(): File {
        val candidates = buildList<File> {
            System.getProperty("CONTROLPLANE_ROOT")?.let { add(File(it)) }
            System.getenv("CONTROLPLANE_ROOT")?.let { add(File(it)) }
            var dir = File(System.getProperty("user.dir")).absoluteFile
            while (true) {
                add(dir.resolve("controlplane"))
                add(dir)
                dir = dir.parentFile ?: break
            }
        }
        for (candidate in candidates) {
            val root = if (File(candidate, "examples").isDirectory) candidate
            else File(candidate, "controlplane").takeIf { File(it, "examples").isDirectory }
            if (root != null) return root.canonicalFile
        }
        error(
            "Cannot locate the repo-level controlplane/ tree. " +
                "Set -DCONTROLPLANE_ROOT=<path to controlplane/> (or its parent). " +
                "Tried: ${candidates.joinToString()}"
        )
    }

    fun examplesDir(): File = File(root(), "examples")

    fun schemasDir(): File = File(root(), "schemas")

    fun example(name: String): File = File(examplesDir(), name)
}
