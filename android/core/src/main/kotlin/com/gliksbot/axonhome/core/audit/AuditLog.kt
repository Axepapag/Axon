package com.gliksbot.axonhome.core.audit

import kotlinx.serialization.Serializable

/**
 * Operational audit trail (mission §14). Every operational mutation
 * initiated through Axon Home records who, device, command, target,
 * timestamp, previous state, requested state, result, and error.
 */
@Serializable
data class AuditEntry(
    val who: String,
    val device: String,
    val command: String,
    val target: String,
    val timestamp: String,
    @kotlinx.serialization.SerialName("previous_state") val previousState: String,
    @kotlinx.serialization.SerialName("requested_state") val requestedState: String,
    val result: String,
    val error: String? = null,
)

/** In-memory audit log; append-only, ordered. */
class AuditLog {
    private val _entries = mutableListOf<AuditEntry>()

    val entries: List<AuditEntry> get() = _entries.toList()

    fun record(entry: AuditEntry): AuditEntry {
        _entries.add(entry)
        return entry
    }

    fun record(
        who: String,
        device: String,
        command: String,
        target: String,
        timestamp: String,
        previousState: String,
        requestedState: String,
        result: String,
        error: String? = null,
    ): AuditEntry = record(
        AuditEntry(who, device, command, target, timestamp, previousState, requestedState, result, error)
    )

    fun forTarget(target: String): List<AuditEntry> = _entries.filter { it.target == target }
}
