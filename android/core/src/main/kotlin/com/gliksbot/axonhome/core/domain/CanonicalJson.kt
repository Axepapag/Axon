package com.gliksbot.axonhome.core.domain

import java.security.MessageDigest

/**
 * Deterministic JSON canonicalization used for every Axon content id.
 *
 * Mirrors `runtime/field/schema.py::canonical_json_bytes` in Axon main:
 *  - object keys sorted lexicographically
 *  - no insignificant whitespace (separators "," and ":")
 *  - strings emitted as raw UTF-8 (ensure_ascii=False semantics): only the
 *    JSON-mandated escapes are produced — `"`, `\`, and control characters
 *    (< 0x20) as \b \t \n \f \r or \u00xx (lowercase hex, like CPython).
 *  - non-finite doubles are rejected (allow_nan=False semantics)
 *  - doubles render with the shortest round-trip form; integral doubles
 *    render with a trailing ".0" (e.g. 1.0), matching Python repr.
 *
 * The canonical byte string is then SHA-256 hashed (hex) to derive ids such
 * as field_id, delta_id, proposal_id, verdict_id, soul_id, receipt_id.
 * This function is the single source of canonical bytes in :core.
 */
object CanonicalJson {

    fun encode(value: Any?): String = buildString { write(value) }

    fun sha256Hex(value: Any?): String =
        sha256Hex(encode(value).encodeToByteArray())

    fun sha256Hex(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
        return digest.joinToString("") { "%02x".format(it) }
    }

    private fun StringBuilder.write(value: Any?) {
        when (value) {
            null -> append("null")
            is Boolean -> append(if (value) "true" else "false")
            is Int -> append(value)
            is Long -> append(value)
            is Double -> writeDouble(value)
            is Float -> writeDouble(value.toDouble())
            is String -> writeString(value)
            is Map<*, *> -> writeObject(value)
            is List<*> -> writeArray(value)
            is Array<*> -> writeArray(value.toList())
            else -> throw IllegalArgumentException(
                "value of type ${value::class.qualifiedName} is not JSON-canonicalizable"
            )
        }
    }

    private fun StringBuilder.writeDouble(value: Double) {
        require(!value.isNaN() && !value.isInfinite()) {
            "non-finite doubles are not permitted in canonical JSON"
        }
        append(value.toString())
    }

    private fun StringBuilder.writeObject(map: Map<*, *>) {
        append('{')
        val keys = map.keys.map { key ->
            require(key is String) { "canonical JSON object keys must be strings" }
            key
        }.sorted()
        keys.forEachIndexed { index, key ->
            if (index > 0) append(',')
            writeString(key)
            append(':')
            write(map[key])
        }
        append('}')
    }

    private fun StringBuilder.writeArray(list: List<*>) {
        append('[')
        list.forEachIndexed { index, item ->
            if (index > 0) append(',')
            write(item)
        }
        append(']')
    }

    private fun StringBuilder.writeString(value: String) {
        append('"')
        for (ch in value) {
            when (ch) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\t' -> append("\\t")
                '\n' -> append("\\n")
                '\u000C' -> append("\\f")
                '\r' -> append("\\r")
                else -> if (ch < ' ') {
                    append("\\u")
                    append(ch.code.toString(16).padStart(4, '0'))
                } else {
                    append(ch)
                }
            }
        }
        append('"')
    }
}

/** Convenience alias mirroring the Python helper name. */
fun canonicalSha256(value: Any?): String = CanonicalJson.sha256Hex(value)
