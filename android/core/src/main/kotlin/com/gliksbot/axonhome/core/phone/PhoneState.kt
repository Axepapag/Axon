package com.gliksbot.axonhome.core.phone

import com.gliksbot.axonhome.core.domain.*
import kotlinx.serialization.Serializable

/** Phone foundation: no synthetic proposals, trained weights, or background timer. */
@Serializable
data class PhoneState(
    val schema: String = "axon-phone-foundation-v1",
    val field: SharedFieldSnapshot,
    val masks: Map<String, Int> = CANONICAL_REGION_ORDER.associate { it.wireName to 100 },
    val ingressCount: Long = 0,
    val paused: Boolean = false,
) {
    init {
        require(schema == "axon-phone-foundation-v1")
        require(ingressCount >= 0)
        require(masks.keys == CANONICAL_REGION_ORDER.map { it.wireName }.toSet())
        require(masks.values.all { it in 0..100 })
        require(masks[LogicalRegion.IDENTITY.wireName] == 100)
        field.regions.forEach { region ->
            requireExactUnicode(region.text)
            // This foundation has no foreign floating-point hash compatibility.
            // Reject imported forms outside the locally supported canonical subset.
            require(region.spans.all { it.confidence == 1.0 })
            require(region.attendedIntervals == null && region.maskPolicy == null)
            require(region.visibility == RegionVisibility.ATTENDED)
        }
    }
}

fun requireExactUnicode(text: String) {
    var i = 0
    while (i < text.length) {
        val c = text[i++]
        if (Character.isHighSurrogate(c)) {
            require(i < text.length && Character.isLowSurrogate(text[i])) { "Unpaired Unicode surrogate" }
            i++
        } else require(!Character.isLowSurrogate(c)) { "Unpaired Unicode surrogate" }
    }
}

object PhoneHeart {
    fun ingress(state: PhoneState, text: String, ingressId: String): PhoneState {
        check(!state.paused) { "Heart is paused" }
        require(text.isNotBlank()) { "Enter a message" }
        requireExactUnicode(text)
        require(ingressId.isNotBlank())
        val current = state.field.region(LogicalRegion.USER_INPUT)
        val span = FieldSpan(ingressId, text, source = "android-user", provenance = ingressId)
        val updated = current.withSpans(current.spans + span)
        val next = state.field.copy(
            parentFieldId = state.field.fieldId,
            regions = state.field.regions.map { if (it.name == updated.name) updated else it },
            // Ingress commits are not reasoning ticks. No Core ran.
        )
        return state.copy(field = next, ingressCount = Math.addExact(state.ingressCount, 1))
    }

    fun mask(state: PhoneState, region: LogicalRegion, percent: Int): PhoneState {
        require(percent in 0..100)
        require(region != LogicalRegion.IDENTITY || percent == 100) { "Identity is always attended" }
        return state.copy(masks = state.masks + (region.wireName to percent))
    }
}

/** No mobile backend is activated until export and full-field coverage are verified. */
interface CoreInferenceBackend {
    val coreId: String
    val architectureId: String
    val candidateGeneration: String
    val checkpointSha256: String
    suspend fun propose(request: CorePassRequest): CorePassResult
}

data class CorePassRequest(
    val field: SharedFieldSnapshot,
    val masks: Map<String, Int>,
    val phase: String,
    val proposalBoard: List<EnglishProposal>,
    val privateSoul: ByteArray,
)

data class CorePassResult(
    val text: String,
    val privateSoul: ByteArray,
    val coverageReceipt: String,
    val terminatedByEos: Boolean,
)

/** This is a body-only recovery envelope, never a full-organism or Core capsule. */
@Serializable
data class PhoneRecovery(
    val schema: String = "axon-phone-body-recovery-v1",
    val sourceCommit: String,
    val payload: String,
    val sha256: String,
    val complete: Boolean = true,
) {
    fun verify(): PhoneState {
        require(schema == "axon-phone-body-recovery-v1" && complete)
        require(sourceCommit.isNotBlank())
        require(CanonicalJson.sha256Hex(payload.encodeToByteArray()) == sha256) { "Recovery checksum mismatch" }
        return AxonJson.decodeFromString(PhoneState.serializer(), payload)
    }
    companion object {
        fun create(state: PhoneState, sourceCommit: String): PhoneRecovery {
            val payload = AxonJson.encodeToString(PhoneState.serializer(), state)
            return PhoneRecovery(sourceCommit = sourceCommit, payload = payload,
                sha256 = CanonicalJson.sha256Hex(payload.encodeToByteArray()))
        }
    }
}
