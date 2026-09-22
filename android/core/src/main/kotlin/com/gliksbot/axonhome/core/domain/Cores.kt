package com.gliksbot.axonhome.core.domain

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Core registry + anatomy contracts — mirrors runtime/heart/registry.py and
 * training/living_reasoning_d64.py LivingReasoningCoreConfig.
 */

@Serializable
enum class CoreStatus {
    @SerialName("active") ACTIVE,
    @SerialName("offline_training") OFFLINE_TRAINING,
    @SerialName("disabled") DISABLED,
}

/**
 * Learned anatomy configuration. d_model is hard-locked to 64; the
 * architecture id is "living-d64-english-" + sha256(config)[:24].
 */
@Serializable
data class AnatomyConfig(
    @SerialName("d_model") val dModel: Int = 64,
    @SerialName("n_heads") val nHeads: Int,
    @SerialName("n_layers") val nLayers: Int,
    @SerialName("ffn_dim") val ffnDim: Int,
    @SerialName("state_tokens") val stateTokens: Int,
    @SerialName("page_size") val pageSize: Int,
) {
    init {
        require(dModel == D_MODEL_LOCK) { "d_model is hard-locked to $D_MODEL_LOCK" }
        require(nHeads >= 1) { "n_heads must be positive" }
        require(nLayers >= 1) { "n_layers must be positive" }
        require(ffnDim >= 1) { "ffn_dim must be positive" }
        require(stateTokens >= 1) { "state_tokens must be positive" }
        require(pageSize >= 1) { "page_size must be positive" }
    }

    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "d_model" to dModel,
        "n_heads" to nHeads,
        "n_layers" to nLayers,
        "ffn_dim" to ffnDim,
        "state_tokens" to stateTokens,
        "page_size" to pageSize,
    )

    val architectureId: String
        get() = ARCHITECTURE_ID_PREFIX + canonicalSha256(toCanonicalDict()).take(24)

    companion object {
        const val D_MODEL_LOCK = 64
        const val ARCHITECTURE_ID_PREFIX = "living-d64-english-"
    }
}

@Serializable
data class CoreDescriptor(
    @SerialName("core_id") val coreId: String,
    @SerialName("d_model") val dModel: Int = AnatomyConfig.D_MODEL_LOCK,
    val status: CoreStatus = CoreStatus.ACTIVE,
    @SerialName("writable_regions") val writableRegions: List<LogicalRegion> =
        CORE_WRITABLE_REGIONS.toList(),
    @SerialName("architecture_id") val architectureId: String = "untrained-reasoning-core-v1",
    @SerialName("parameter_generation") val parameterGeneration: String = "untrained",
) {
    init {
        require(coreId.isNotEmpty()) { "CoreDescriptor.core_id must be non-empty" }
        require(dModel == AnatomyConfig.D_MODEL_LOCK) { "only d_model=64 cores are permitted" }
    }
}

@Serializable
data class CoreBinding(
    @SerialName("architecture_id") val architectureId: String,
    @SerialName("parameter_generation") val parameterGeneration: String,
    @SerialName("optimizer_generation") val optimizerGeneration: String,
    @SerialName("soul_id") val soulId: String,
    val mode: String,
)
