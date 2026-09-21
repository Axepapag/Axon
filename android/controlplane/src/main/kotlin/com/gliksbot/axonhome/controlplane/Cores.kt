package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** CoreRegistry descriptor + binding view (arch §5.5, recon B §6). Read-only. */
@Serializable
data class CoreDescriptor(
    @SerialName("core_id") val coreId: String,
    /** Hard-locked D64 (LivingReasoningCoreConfig). */
    @SerialName("d_model") val dModel: Int = 64,
    val status: CoreStatus,
    /** e.g. living-d64-english-<sha256(config)[:24]>, or untrained-reasoning-core-v1. */
    @SerialName("architecture_id") val architectureId: String,
    /** "untrained" or a candidate/base generation id. */
    @SerialName("parameter_generation") val parameterGeneration: String,
    /** Only scratch and response_draft are core-writable. */
    @SerialName("writable_regions") val writableRegions: List<LogicalRegion> = emptyList(),
    @SerialName("soul_id") val soulId: String? = null,
    @SerialName("soul_generation") val soulGeneration: Long? = null,
    @SerialName("optimizer_generation") val optimizerGeneration: String? = null,
    val mode: String? = null,
)

@Serializable
enum class CoreStatus {
    @SerialName("active") ACTIVE,
    @SerialName("offline_training") OFFLINE_TRAINING,
    @SerialName("disabled") DISABLED,
}

/** GET /v1/cores. */
@Serializable
data class CoreList(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val cores: List<CoreDescriptor>,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-core-list-v1"
    }
}

/** GET /v1/cores/{coreId}. */
@Serializable
data class CoreDetail(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val core: CoreDescriptor,
    /** Living core anatomy when architecture is a LivingReasoningCoreConfig. */
    val anatomy: CoreAnatomy? = null,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-core-detail-v1"
    }
}

/** Locked anatomy (recon B §6): d_model=64, n_heads=1, n_layers=2, ffn_dim=131072, state_tokens=4, page_size=32. */
@Serializable
data class CoreAnatomy(
    @SerialName("d_model") val dModel: Int = 64,
    @SerialName("n_heads") val nHeads: Int,
    @SerialName("n_layers") val nLayers: Int,
    @SerialName("ffn_dim") val ffnDim: Int,
    @SerialName("state_tokens") val stateTokens: Int,
    @SerialName("page_size") val pageSize: Int,
)
