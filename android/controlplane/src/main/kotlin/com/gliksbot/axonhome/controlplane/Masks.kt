package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Masks are a DERIVED VIEW, never canonical state (arch §2.4). Durable record:
 * axon-heart-region-masks-v3 at active/heart/region_masks.json. Changing masks
 * advances the revision/state_id and produces a new view_id.
 */
@Serializable
data class MaskPolicy(
    val kind: MaskKind,
    /** Newest-suffix slider 0–100; required iff kind == TAIL_PERCENT. */
    @SerialName("tail_percent") val tailPercent: Int? = null,
    /** Required iff kind == LAST_N_SPANS. */
    val n: Int? = null,
)

@Serializable
enum class MaskKind {
    @SerialName("all") ALL,
    @SerialName("none") NONE,
    @SerialName("last_n_spans") LAST_N_SPANS,
    @SerialName("tail_percent") TAIL_PERCENT,
}

/** GET /v1/masks response; also the proposed/resulting state in mask updates. */
@Serializable
data class MaskState(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    /** Durable record schema. */
    @SerialName("mask_schema") val maskSchema: String = MASK_SCHEMA,
    /** Monotonic mask revision (axon-heart-region-masks-v3). */
    val revision: Long,
    @SerialName("state_id") val stateId: String,
    /** derive_view_id(masks). */
    @SerialName("view_id") val viewId: String,
    /** All 13 canonical regions present; identity is always ALL. */
    val policies: Map<LogicalRegion, MaskPolicy>,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-mask-state-v1"
        const val MASK_SCHEMA = "axon-heart-region-masks-v3"
    }
}

/**
 * PUT /v1/masks request. Confirm semantics are mandatory (arch §5.5): the first
 * PUT (confirmed=false) returns a proposal + confirm_token and changes nothing;
 * a second PUT with confirmToken + confirmed=true applies via
 * HeartHost.set_region_mask_policy.
 */
@Serializable
data class PutMasksRequest(
    val schema: String = SCHEMA,
    val policies: Map<LogicalRegion, MaskPolicy>,
    @SerialName("confirm_token") val confirmToken: String? = null,
    val confirmed: Boolean = false,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-mask-update-v1"
    }
}

@Serializable
data class PutMasksResponse(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    /** false for a proposal (first touch); true only after a confirmed second PUT. */
    val applied: Boolean,
    /** Present on a proposal; single-use, short-lived. */
    @SerialName("confirm_token") val confirmToken: String? = null,
    val proposed: MaskState,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-mask-update-result-v1"
    }
}
