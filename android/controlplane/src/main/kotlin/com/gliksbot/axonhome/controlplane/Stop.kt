package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Four distinct stops, each honest about its boundary (arch §13):
 * - PAUSE_AXON: finish/abort the current beat at the atomic runtime boundary (Heart host BeatState idle).
 * - STOP_TRAINING: stop at a recoverable Trainer tranche boundary. A mid-segment
 *   interactive pause is a FUTURE TrainerOrgan handler; until wired the organ
 *   returns UNAVAILABLE and the ack says exactly that.
 * - STOP_COMPUTE: terminate a selected worker/provider job (only when the
 *   provider advertises supportsStop).
 * - EMERGENCY: attempt to halt organism execution + training immediately with
 *   full ack accounting.
 */
@Serializable
enum class StopScope {
    @SerialName("pause_axon") PAUSE_AXON,
    @SerialName("stop_training") STOP_TRAINING,
    @SerialName("stop_compute") STOP_COMPUTE,
    @SerialName("emergency") EMERGENCY,
}

@Serializable
enum class ComponentAckStatus {
    @SerialName("confirmed_stopped") CONFIRMED_STOPPED,
    @SerialName("unreachable") UNREACHABLE,
    @SerialName("provably_gone") PROVABLY_GONE,
    @SerialName("refused") REFUSED,
    @SerialName("unavailable") UNAVAILABLE,
}

@Serializable
enum class ComponentKind {
    @SerialName("heart_host") HEART_HOST,
    @SerialName("trainer") TRAINER,
    @SerialName("compute_worker") COMPUTE_WORKER,
    @SerialName("control_plane") CONTROL_PLANE,
}

@Serializable
data class ComponentAck(
    @SerialName("component_id") val componentId: String,
    @SerialName("component_kind") val componentKind: ComponentKind,
    val status: ComponentAckStatus,
    /** Verbatim upstream answer where applicable, e.g. Trainer UNAVAILABLE message. */
    val detail: String? = null,
)

/** POST /v1/stop request. */
@Serializable
data class StopRequest(
    val schema: String = SCHEMA,
    val scope: StopScope,
    /** Required for STOP_COMPUTE (worker id); optional elsewhere. */
    val target: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-stop-request-v1"
    }
}

/**
 * POST /v1/stop response: a per-component ack map. [allStopped] is true ONLY
 * when every registered component acknowledged or is provably gone — see
 * [StopAccounting]. Never fakes successful cancellation (arch §13).
 */
@Serializable
data class StopResponse(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val scope: StopScope,
    @SerialName("all_stopped") val allStopped: Boolean,
    val components: List<ComponentAck>,
    @SerialName("audit_id") val auditId: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-stop-result-v1"
    }
}

/**
 * Honest stop-ack accounting (arch §13). `allStopped` may be true iff every
 * component is CONFIRMED_STOPPED or PROVABLY_GONE. Any UNREACHABLE (or REFUSED /
 * UNAVAILABLE) component forces allStopped=false.
 *
 * Server implementers MUST compute allStopped with [computeAllStopped] (or
 * equivalent logic); clients SHOULD re-derive it with [verify] and distrust any
 * response whose claim disagrees with the ack map.
 */
object StopAccounting {
    fun computeAllStopped(components: List<ComponentAck>): Boolean =
        components.isNotEmpty() && components.all {
            it.status == ComponentAckStatus.CONFIRMED_STOPPED ||
                it.status == ComponentAckStatus.PROVABLY_GONE
        }

    /** Components that block an "everything stopped" claim. */
    fun unresolved(components: List<ComponentAck>): List<ComponentAck> =
        components.filter {
            it.status != ComponentAckStatus.CONFIRMED_STOPPED &&
                it.status != ComponentAckStatus.PROVABLY_GONE
        }

    /** True when the response's allStopped claim matches the ack map. */
    fun verify(response: StopResponse): Boolean =
        response.allStopped == computeAllStopped(response.components)
}
