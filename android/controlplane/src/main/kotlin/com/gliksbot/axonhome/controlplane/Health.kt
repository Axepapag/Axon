package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** GET /v1/health — Control Plane liveness + distilled Heart health (axon-heart-health-v2). */
@Serializable
data class HealthResponse(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    @SerialName("control_plane") val controlPlane: ControlPlaneStatus,
    /** Null when the Heart host is unreachable. */
    val heart: HeartHealthSummary?,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-health-v1"
    }
}

@Serializable
data class ControlPlaneStatus(
    val status: String, // "ok" | "degraded"
    val version: String,
    @SerialName("api_versions") val apiVersions: List<String>,
    @SerialName("uptime_seconds") val uptimeSeconds: Long? = null,
)

/** Distilled axon-heart-health-v2 (active/heart/health_latest.json). */
@Serializable
data class HeartHealthSummary(
    val schema: String = SCHEMA,
    @SerialName("heart_epoch_id") val heartEpochId: String,
    @SerialName("heartbeat_sequence") val heartbeatSequence: Long,
    @SerialName("tick_sequence") val tickSequence: Long,
    @SerialName("head_field_id") val headFieldId: String? = null,
    @SerialName("tick_in_flight") val tickInFlight: Boolean,
    @SerialName("last_beat_ok") val lastBeatOk: Boolean,
    @SerialName("last_failure") val lastFailure: String? = null,
    @SerialName("mask_revision") val maskRevision: Long? = null,
    @SerialName("last_view_id") val lastViewId: String? = null,
    @SerialName("dormant_index_id") val dormantIndexId: String? = null,
    @SerialName("ingress_quarantine_count") val ingressQuarantineCount: Long? = null,
    @SerialName("ingress_rejection_count") val ingressRejectionCount: Long? = null,
) {
    companion object {
        const val SCHEMA = "axon-heart-health-v2"
    }
}
