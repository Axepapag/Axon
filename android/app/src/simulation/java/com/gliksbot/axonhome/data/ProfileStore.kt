package com.gliksbot.axonhome.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

/** Control Plane endpoint profile (arch §10.T). Never carries secrets. */
@Serializable
data class EndpointProfile(
    val id: String,
    val name: String,
    val baseUrl: String,
    /** REFERENCE into the Secrets Vault; never a raw token. */
    val vaultKeyRef: String? = null,
    val isSimulation: Boolean = false,
)

/** Agent profile (arch §10.R). apiKeyRef is a vault REFERENCE, never raw. */
@Serializable
data class AgentProfile(
    val id: String,
    val displayName: String,
    val provider: String,
    val baseUrl: String = "",
    val model: String = "",
    val vaultKeyRef: String? = null,
    val systemInstructions: String = "",
    val allowCompute: Boolean = false,
    val allowStorage: Boolean = false,
    val allowRepo: Boolean = false,
    val timeoutSeconds: Int = 60,
    val avatarColor: Long = 0xFF7C4DFF,
    val active: Boolean = true,
)

@Serializable
private data class ProfileFile(
    val endpoints: List<EndpointProfile> = emptyList(),
    val agents: List<AgentProfile> = emptyList(),
)

/**
 * Local, non-secret profile store: JSON in filesDir (arch §4.3 — cache is a
 * mirror, never an authority). Secrets are only ever vault references.
 */
class ProfileStore(context: Context) {

    private val file = File(context.filesDir, "profiles.json")
    private val json = Json { prettyPrint = true; ignoreUnknownKeys = true }
    private val mutex = Mutex()

    private fun read(): ProfileFile = runCatching {
        if (file.exists()) json.decodeFromString<ProfileFile>(file.readText()) else ProfileFile()
    }.getOrElse { ProfileFile() }

    private fun write(value: ProfileFile) {
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(json.encodeToString(value))
        tmp.renameTo(file)
    }

    suspend fun endpoints(): List<EndpointProfile> = withContext(Dispatchers.IO) {
        mutex.withLock {
            val stored = read().endpoints
            val sim = EndpointProfile(
                id = "sim",
                name = "Built-in simulator",
                baseUrl = "sim://local",
                isSimulation = true,
            )
            listOf(sim) + stored
        }
    }

    suspend fun saveEndpoint(profile: EndpointProfile): Unit = withContext(Dispatchers.IO) {
        mutex.withLock {
            val current = read()
            val next = current.endpoints.filterNot { it.id == profile.id } + profile.copy(isSimulation = false)
            write(current.copy(endpoints = next))
        }
    }

    suspend fun deleteEndpoint(id: String): Unit = withContext(Dispatchers.IO) {
        mutex.withLock {
            val current = read()
            write(current.copy(endpoints = current.endpoints.filterNot { it.id == id }))
        }
    }

    suspend fun agents(): List<AgentProfile> = withContext(Dispatchers.IO) {
        mutex.withLock { read().agents }
    }

    suspend fun saveAgent(profile: AgentProfile): Unit = withContext(Dispatchers.IO) {
        mutex.withLock {
            val current = read()
            val next = current.agents.filterNot { it.id == profile.id } + profile
            write(current.copy(agents = next))
        }
    }

    suspend fun deleteAgent(id: String): Unit = withContext(Dispatchers.IO) {
        mutex.withLock {
            val current = read()
            write(current.copy(agents = current.agents.filterNot { it.id == id }))
        }
    }
}
