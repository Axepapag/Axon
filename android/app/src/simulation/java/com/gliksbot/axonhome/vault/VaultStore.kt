package com.gliksbot.axonhome.vault

import android.content.Context
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

/** Vault entry metadata. The value is stored sealed; only [maskedHint] is ever shown. */
@Serializable
data class VaultEntryMeta(
    val id: String,
    val label: String,
    val scope: String,
    val createdAtEpochSeconds: Long,
    var lastUsedEpochSeconds: Long? = null,
    /** Last-4 hint only, e.g. "••••3f9a". Never the full value. */
    val maskedHint: String,
    /** Base64 of the Keystore-sealed blob (IV || ciphertext || tag). */
    val sealedValue: String,
)

@Serializable
private data class VaultFile(val entries: List<VaultEntryMeta> = emptyList())

/** Mask a secret for display: at most the last 4 characters are ever shown. */
fun maskSecret(value: String): String =
    "••••" + value.takeLast(4).ifEmpty { "∅" }

/**
 * Keystore-sealed secret store (foundation scope, arch §8): metadata JSON in
 * filesDir + per-entry AES-GCM blobs sealed by AndroidKeyStore keys.
 */
class VaultStore(
    context: Context,
    private val box: KeystoreAesGcmBox = KeystoreAesGcmBox(),
) {
    private val file = File(context.filesDir, "vault.json")
    private val json = Json { prettyPrint = true; ignoreUnknownKeys = true }
    private val mutex = Mutex()

    private fun keyAlias(id: String) = "axon_vault_$id"

    private fun read(): VaultFile = runCatching {
        if (file.exists()) json.decodeFromString<VaultFile>(file.readText()) else VaultFile()
    }.getOrElse { VaultFile() }

    private fun write(value: VaultFile) {
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(json.encodeToString(value))
        tmp.renameTo(file)
    }

    suspend fun entries(): List<VaultEntryMeta> = withContext(Dispatchers.IO) {
        mutex.withLock { read().entries }
    }

    suspend fun add(label: String, scope: String, secret: String): VaultEntryMeta =
        withContext(Dispatchers.IO) {
            mutex.withLock {
                val id = java.util.UUID.randomUUID().toString()
                val sealed = box.seal(keyAlias(id), secret.encodeToByteArray())
                val meta = VaultEntryMeta(
                    id = id,
                    label = label,
                    scope = scope,
                    createdAtEpochSeconds = System.currentTimeMillis() / 1000,
                    maskedHint = maskSecret(secret),
                    sealedValue = Base64.encodeToString(sealed, Base64.NO_WRAP),
                )
                val current = read()
                write(current.copy(entries = current.entries + meta))
                meta
            }
        }

    /** Unseal for a copy-to-clipboard action. Caller must have re-authenticated. */
    suspend fun unseal(id: String): String? = withContext(Dispatchers.IO) {
        mutex.withLock {
            val current = read()
            val meta = current.entries.firstOrNull { it.id == id } ?: return@withLock null
            val value = box.unseal(keyAlias(id), Base64.decode(meta.sealedValue, Base64.NO_WRAP))
                .decodeToString()
            val touched = current.entries.map {
                if (it.id == id) it.copy(lastUsedEpochSeconds = System.currentTimeMillis() / 1000) else it
            }
            write(current.copy(entries = touched))
            value
        }
    }

    suspend fun delete(id: String): Unit = withContext(Dispatchers.IO) {
        mutex.withLock {
            val current = read()
            write(current.copy(entries = current.entries.filterNot { it.id == id }))
            box.delete(keyAlias(id))
        }
    }
}
