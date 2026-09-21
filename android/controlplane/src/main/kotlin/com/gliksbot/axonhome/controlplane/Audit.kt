package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
enum class AuditResult {
    @SerialName("success") SUCCESS,
    @SerialName("failure") FAILURE,
    @SerialName("partial") PARTIAL,
    @SerialName("unavailable") UNAVAILABLE,
    @SerialName("rejected") REJECTED,
}

/**
 * Operational audit entry (arch §14). This is NOT the Engineer's Ledger —
 * ledger writes happen only via scripts/append_engineers_ledger_event.py by a
 * participating agent, never from this API.
 */
@Serializable
data class AuditEntry(
    val schema: String = SCHEMA,
    @SerialName("audit_id") val auditId: String,
    /** Authenticated identity. */
    val who: String,
    /** Enrolled device identifier. */
    val device: String,
    /** Endpoint + operation, e.g. "POST /v1/trainer/commands". */
    val command: String,
    /** Affected resource, e.g. region, core id, worker id, capsule id. */
    val target: String? = null,
    val timestamp: Instant,
    @SerialName("previous_state") val previousState: String? = null,
    @SerialName("requested_state") val requestedState: String? = null,
    val result: AuditResult,
    val error: String? = null,
    @SerialName("idempotency_key") val idempotencyKey: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-audit-entry-v1"
    }
}

/** GET /v1/audit. */
@Serializable
data class AuditList(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    /** Newest first. */
    val entries: List<AuditEntry>,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-audit-list-v1"
    }
}
