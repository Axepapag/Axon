package com.gliksbot.axonhome

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.net.URI
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Stores the active Axon host endpoint plus bearer token without asking the
 * operator to paste credentials every launch. The normal phone-sovereign
 * endpoint is loopback (Termux/Python); HTTPS remains valid for an optional
 * remote host. The token is encrypted with a non-exportable Android Keystore
 * key. Clearing app data intentionally removes enrollment.
 */
class RemoteConnectionStore(context: Context) {
    private val prefs = context.getSharedPreferences("axon-home-remote", Context.MODE_PRIVATE)
    private val alias = "axon-home-control-token-v1"

    data class Enrollment(val baseUrl: String, val token: String)

    fun load(): Enrollment? {
        val url = prefs.getString("base_url", null)?.trim()?.trimEnd('/') ?: return null
        val body = prefs.getString("token_ciphertext", null) ?: return null
        val iv = prefs.getString("token_iv", null) ?: return null
        return runCatching { Enrollment(url, decrypt(body, iv)) }.getOrNull()
    }

    fun save(baseUrl: String, token: String) {
        val url = baseUrl.trim().trimEnd('/')
        require(isAllowedEndpoint(url)) {
            "Use HTTPS for remote hosts; HTTP is allowed only on phone loopback"
        }
        require(token.isNotBlank()) { "Control token is required" }
        val (ciphertext, iv) = encrypt(token.trim())
        prefs.edit()
            .putString("base_url", url)
            .putString("token_ciphertext", ciphertext)
            .putString("token_iv", iv)
            .apply()
    }

    fun clear() = prefs.edit().clear().apply()

    private fun isAllowedEndpoint(value: String): Boolean {
        val uri = runCatching { URI(value) }.getOrNull() ?: return false
        val scheme = uri.scheme?.lowercase() ?: return false
        val host = uri.host?.lowercase() ?: return false
        if (scheme == "https") return true
        return scheme == "http" && host in setOf("127.0.0.1", "localhost", "::1")
    }

    private fun secretKey(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey(alias, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build()
        )
        return generator.generateKey()
    }

    private fun encrypt(value: String): Pair<String, String> {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, secretKey())
        val ciphertext = cipher.doFinal(value.encodeToByteArray())
        return Base64.encodeToString(ciphertext, Base64.NO_WRAP) to Base64.encodeToString(cipher.iv, Base64.NO_WRAP)
    }

    private fun decrypt(ciphertext: String, iv: String): String {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, secretKey(), GCMParameterSpec(128, Base64.decode(iv, Base64.NO_WRAP)))
        return cipher.doFinal(Base64.decode(ciphertext, Base64.NO_WRAP)).decodeToString()
    }
}
