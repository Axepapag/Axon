package com.gliksbot.axonhome.core.domain

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Soul contracts — mirrors runtime/soul/contracts.py. Souls are the private
 * learned state of a Core: four opaque temperature layers, integer
 * generations, sha256-chained lineage, and commit receipts.
 */

const val SOUL_LAYER_SCHEMA = "axon-private-soul-layer-v1"
const val SOUL_SNAPSHOT_SCHEMA = "axon-private-soul-snapshot-v1"
const val SOUL_TRANSITION_SCHEMA = "axon-private-soul-transition-v1"
const val SOUL_COMMIT_RECEIPT_SCHEMA = "axon-private-soul-commit-receipt-v1"
const val D64_SOUL_MEDIA_TYPE = "application/x-axon-d64-recurrent-state"
const val D64_SOUL_TENSOR_LAYOUT = "f32le[4,64]"

@Serializable
enum class SoulTemperature {
    @SerialName("hot") HOT,
    @SerialName("warm") WARM,
    @SerialName("cold") COLD,
    @SerialName("deep_cold") DEEP_COLD,
}

/** Fixed order: exactly one layer per temperature. */
val SOUL_TEMPERATURE_ORDER: List<SoulTemperature> = SoulTemperature.entries.toList()

@Serializable
data class SoulLayer(
    val temperature: SoulTemperature,
    @SerialName("media_type") val mediaType: String,
    @SerialName("tensor_layout") val tensorLayout: String,
    @SerialName("payload_bytes") val payloadBytes: Long,
    @SerialName("payload_sha256") val payloadSha256: String,
    @SerialName("payload_base64") val payloadBase64: String? = null,
    @SerialName("schema") val schema: String = SOUL_LAYER_SCHEMA,
) {
    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to SOUL_LAYER_SCHEMA,
        "temperature" to AxonJson.encodeToString(SoulTemperature.serializer(), temperature).trim('"'),
        "media_type" to mediaType,
        "tensor_layout" to tensorLayout,
        "payload_bytes" to payloadBytes,
        "payload_sha256" to payloadSha256,
        "payload_base64" to payloadBase64,
    )
}

@Serializable
data class SoulSnapshot(
    @SerialName("core_id") val coreId: String,
    @SerialName("architecture_id") val architectureId: String,
    @SerialName("parameter_generation") val parameterGeneration: String,
    val generation: Int,
    @SerialName("parent_soul_id") val parentSoulId: String?,
    val layers: List<SoulLayer>,
    @SerialName("schema") val schema: String = SOUL_SNAPSHOT_SCHEMA,
) {
    init {
        require(schema == SOUL_SNAPSHOT_SCHEMA) { "unsupported soul snapshot schema '$schema'" }
        require(generation >= 0) { "soul generation must be non-negative" }
        require(layers.map { it.temperature } == SOUL_TEMPERATURE_ORDER) {
            "soul must carry exactly one layer per temperature in canonical order"
        }
    }

    fun layer(temperature: SoulTemperature): SoulLayer = layers[temperature.ordinal]

    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to SOUL_SNAPSHOT_SCHEMA,
        "core_id" to coreId,
        "architecture_id" to architectureId,
        "parameter_generation" to parameterGeneration,
        "generation" to generation,
        "parent_soul_id" to parentSoulId,
        "layers" to layers.map { it.toCanonicalDict() },
    )

    val soulId: String get() = canonicalSha256(toCanonicalDict())
}

/** Runtime pass phases first/refined/consolidated must update HOT. */
@Serializable
data class SoulTransition(
    @SerialName("core_id") val coreId: String,
    @SerialName("branch_id") val branchId: String,
    @SerialName("before_soul_id") val beforeSoulId: String,
    val phase: String,
    @SerialName("updated_temperatures") val updatedTemperatures: List<SoulTemperature>,
    @SerialName("tick_uid") val tickUid: String? = null,
    @SerialName("request_id") val requestId: String? = null,
    @SerialName("schema") val schema: String = SOUL_TRANSITION_SCHEMA,
) {
    init {
        require(schema == SOUL_TRANSITION_SCHEMA) { "unsupported soul transition schema '$schema'" }
        if (phase in setOf("first", "refined", "consolidated")) {
            require(SoulTemperature.HOT in updatedTemperatures) {
                "runtime phase '$phase' transitions must update the HOT layer"
            }
        }
    }

    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to SOUL_TRANSITION_SCHEMA,
        "core_id" to coreId,
        "branch_id" to branchId,
        "before_soul_id" to beforeSoulId,
        "phase" to phase,
        "updated_temperatures" to updatedTemperatures.map {
            AxonJson.encodeToString(SoulTemperature.serializer(), it).trim('"')
        },
        "tick_uid" to tickUid,
        "request_id" to requestId,
    )

    val transitionId: String get() = canonicalSha256(toCanonicalDict())
}

@Serializable
data class SoulCommitReceipt(
    @SerialName("core_id") val coreId: String,
    @SerialName("branch_id") val branchId: String,
    @SerialName("transition_id") val transitionId: String,
    @SerialName("before_soul_id") val beforeSoulId: String,
    @SerialName("after_soul_id") val afterSoulId: String,
    val generation: Int,
    val phase: String,
    @SerialName("tick_uid") val tickUid: String? = null,
    @SerialName("request_id") val requestId: String? = null,
    @SerialName("commit_binding") val commitBinding: String? = null,
    @SerialName("schema") val schema: String = SOUL_COMMIT_RECEIPT_SCHEMA,
) {
    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to SOUL_COMMIT_RECEIPT_SCHEMA,
        "core_id" to coreId,
        "branch_id" to branchId,
        "transition_id" to transitionId,
        "before_soul_id" to beforeSoulId,
        "after_soul_id" to afterSoulId,
        "generation" to generation,
        "phase" to phase,
        "tick_uid" to tickUid,
        "request_id" to requestId,
        "commit_binding" to commitBinding,
    )

    val receiptId: String get() = canonicalSha256(toCanonicalDict())
}

class SoulTransitionError(message: String) : IllegalStateException(message)

/**
 * Apply a transition to [before]: enforces before-idempotence, rewrites the
 * updated layers' opaque payloads (deterministic content-derived bytes) and
 * advances generation by exactly 1. Returns the successor snapshot and the
 * commit receipt chaining before→after.
 */
fun applySoulTransition(
    before: SoulSnapshot,
    transition: SoulTransition,
    commitBinding: String? = null,
    payloadDerivation: (SoulTemperature, SoulSnapshot) -> ByteArray = { temperature, soul ->
        // Deterministic opaque payload: derived from the before-soul hash, the
        // temperature and the transition id. Doctrine-opaque bytes.
        CanonicalJson.sha256Hex("${soul.soulId}|${transition.transitionId}|$temperature")
            .encodeToByteArray()
    },
): Pair<SoulSnapshot, SoulCommitReceipt> {
    if (transition.coreId != before.coreId) {
        throw SoulTransitionError("transition core '${transition.coreId}' does not own this soul")
    }
    if (transition.beforeSoulId != before.soulId) {
        throw SoulTransitionError(
            "transition before_soul_id ${transition.beforeSoulId} does not match current soul ${before.soulId}"
        )
    }
    val nextLayers = before.layers.map { layer ->
        if (layer.temperature in transition.updatedTemperatures) {
            val payload = payloadDerivation(layer.temperature, before)
            layer.copy(
                payloadBytes = payload.size.toLong(),
                payloadSha256 = CanonicalJson.sha256Hex(payload),
                payloadBase64 = null,
            )
        } else {
            layer
        }
    }
    val after = SoulSnapshot(
        coreId = before.coreId,
        architectureId = before.architectureId,
        parameterGeneration = before.parameterGeneration,
        generation = before.generation + 1,
        parentSoulId = before.soulId,
        layers = nextLayers,
    )
    val receipt = SoulCommitReceipt(
        coreId = before.coreId,
        branchId = transition.branchId,
        transitionId = transition.transitionId,
        beforeSoulId = before.soulId,
        afterSoulId = after.soulId,
        generation = after.generation,
        phase = transition.phase,
        tickUid = transition.tickUid,
        requestId = transition.requestId,
        commitBinding = commitBinding,
    )
    return after to receipt
}

/** Initial genesis soul for a core (generation 0, no parent). */
fun genesisSoul(
    coreId: String,
    architectureId: String,
    parameterGeneration: String,
    seed: String,
): SoulSnapshot = SoulSnapshot(
    coreId = coreId,
    architectureId = architectureId,
    parameterGeneration = parameterGeneration,
    generation = 0,
    parentSoulId = null,
    layers = SOUL_TEMPERATURE_ORDER.map { temperature ->
        val payload = CanonicalJson.sha256Hex("genesis|$seed|$coreId|$temperature").encodeToByteArray()
        SoulLayer(
            temperature = temperature,
            mediaType = D64_SOUL_MEDIA_TYPE,
            tensorLayout = D64_SOUL_TENSOR_LAYOUT,
            payloadBytes = payload.size.toLong(),
            payloadSha256 = CanonicalJson.sha256Hex(payload),
        )
    },
)
