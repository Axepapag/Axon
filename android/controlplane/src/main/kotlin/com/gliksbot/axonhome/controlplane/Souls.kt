package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * GET /v1/souls/{coreId} — Soul Observatory metadata (arch §10.H).
 * METADATA ONLY: soul payloads are doctrine-opaque; only ids, generations,
 * sizes, hashes and receipt chains are exposed.
 */
@Serializable
data class SoulSummary(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    @SerialName("core_id") val coreId: String,
    /** Soul branch, typically "live". */
    @SerialName("branch_id") val branchId: String,
    val head: SoulHead,
    /** Exactly four layers, one per temperature, fixed order hot/warm/cold/deep_cold. */
    val layers: List<SoulLayerMeta>,
    /** Tail of the exact linear receipt chain. */
    @SerialName("receipts_tail") val receiptsTail: List<SoulCommitReceiptMeta>,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-soul-summary-v1"
    }
}

/** Soul branch HEAD (durable record schema axon-private-soul-head-v1). */
@Serializable
data class SoulHead(
    val schema: String = SCHEMA,
    @SerialName("soul_id") val soulId: String,
    val generation: Long,
    @SerialName("parent_soul_id") val parentSoulId: String? = null,
    @SerialName("parameter_generation") val parameterGeneration: String? = null,
    @SerialName("architecture_id") val architectureId: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-private-soul-head-v1"
    }
}

@Serializable
enum class SoulTemperature {
    @SerialName("hot") HOT,
    @SerialName("warm") WARM,
    @SerialName("cold") COLD,
    @SerialName("deep_cold") DEEP_COLD,
}

/** axon-private-soul-layer-v1 metadata. Payload bytes are never exposed. */
@Serializable
data class SoulLayerMeta(
    val temperature: SoulTemperature,
    /** e.g. application/x-axon-d64-recurrent-state (codec axon-d64-recurrent-soul-codec-v1). */
    @SerialName("media_type") val mediaType: String,
    @SerialName("payload_bytes") val payloadBytes: Long,
    @SerialName("payload_sha256") val payloadSha256: String,
    /** Always true: payloads are never interpreted by this API. */
    val opaque: Boolean = true,
)

@Serializable
enum class SoulPhase {
    @SerialName("first") FIRST,
    @SerialName("refined") REFINED,
    @SerialName("consolidated") CONSOLIDATED,
    @SerialName("outcome") OUTCOME,
    @SerialName("training") TRAINING,
    @SerialName("merge") MERGE,
}

/** axon-private-soul-commit-receipt-v1. */
@Serializable
data class SoulCommitReceiptMeta(
    val schema: String = SCHEMA,
    @SerialName("receipt_id") val receiptId: String,
    @SerialName("transition_id") val transitionId: String? = null,
    @SerialName("before_soul_id") val beforeSoulId: String,
    @SerialName("after_soul_id") val afterSoulId: String,
    val generation: Long,
    val phase: SoulPhase,
    @SerialName("tick_uid") val tickUid: String? = null,
    /** e.g. canonical-field:{successor.field_id} for the consolidator. */
    @SerialName("commit_binding") val commitBinding: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-private-soul-commit-receipt-v1"
    }
}
