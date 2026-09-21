package com.gliksbot.axonhome.vault

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * AndroidKeyStore-backed AES-GCM box (arch §8). One AES-GCM key per vault
 * entry alias, generated inside secure hardware; key material never leaves
 * the Keystore. No custom cryptography — platform Cipher only.
 *
 * Layout of sealed blobs: 12-byte IV || ciphertext || 16-byte GCM tag.
 */
class KeystoreAesGcmBox(
    private val keyStore: KeyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) },
) {
    private fun key(alias: String): SecretKey {
        keyStore.getKey(alias, null)?.let { return it as SecretKey }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(
                alias,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .setRandomizedEncryptionRequired(true)
                // Note: setUserAuthenticationRequired is deliberately NOT set
                // on the key; unlock UX is enforced at the app layer (biometric
                // prompt before unseal) so keys stay usable with device-credential
                // fallback on older devices. Documented foundation scope.
                .build()
        )
        return generator.generateKey()
    }

    fun seal(alias: String, plaintext: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, key(alias))
        val ciphertext = cipher.doFinal(plaintext)
        return cipher.iv + ciphertext
    }

    fun unseal(alias: String, sealed: ByteArray): ByteArray {
        require(sealed.size > 12) { "sealed blob too short" }
        val iv = sealed.copyOfRange(0, 12)
        val ciphertext = sealed.copyOfRange(12, sealed.size)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, key(alias), GCMParameterSpec(128, iv))
        return cipher.doFinal(ciphertext)
    }

    fun delete(alias: String) {
        if (keyStore.containsAlias(alias)) keyStore.deleteEntry(alias)
    }
}
