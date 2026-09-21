package com.gliksbot.axonhome.core.domain

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Heart health + region mask controller state — mirrors runtime/heart/
 * health.py (axon-heart-health-v2, subset relevant to the app) and masks.py
 * (axon-heart-region-masks-v3).
 */

const val HEART_HEALTH_SCHEMA = "axon-heart-health-v2"
const val REGION_MASKS_SCHEMA = "axon-heart-region-masks-v3"

@Serializable
data class HeartHealth(
    @SerialName("epoch_id") val epochId: String,
    @SerialName("heartbeat_sequence") val heartbeatSequence: Long,
    @SerialName("tick_sequence") val tickSequence: Long,
    @SerialName("head_field_id") val headFieldId: String,
    @SerialName("head_tick_id") val headTickId: Int,
    @SerialName("tick_in_flight") val tickInFlight: Boolean,
    @SerialName("mask_revision") val maskRevision: Long,
    @SerialName("mask_state_id") val maskStateId: String,
    @SerialName("last_view_id") val lastViewId: String,
    @SerialName("last_failure") val lastFailure: String? = null,
    @SerialName("schema") val schema: String = HEART_HEALTH_SCHEMA,
)

/**
 * Durable mask controller state. Monotonic [revision]; [stateId] is derived
 * from revision + policies. Masks are a DERIVED VIEW — changing them never
 * mutates the canonical field. Region keys are canonical region wire names.
 */
@Serializable
data class HeartRegionMaskState(
    val revision: Long,
    @SerialName("state_id") val stateId: String,
    val policies: Map<String, RegionMaskPolicy>,
    @SerialName("schema") val schema: String = REGION_MASKS_SCHEMA,
) {
    init {
        require(schema == REGION_MASKS_SCHEMA) { "unsupported mask state schema '$schema'" }
        require(revision >= 0) { "mask revision must be non-negative" }
    }

    fun policyFor(region: LogicalRegion): RegionMaskPolicy =
        policies[region.wireName] ?: defaultPolicyFor(region)

    /** Exact attended intervals for [region] against its current spans. */
    fun resolve(region: LogicalRegion, spans: List<FieldSpan>): List<AttendedInterval> =
        resolveMaskPolicy(spans, policyFor(region))

    fun withPolicy(region: LogicalRegion, policy: RegionMaskPolicy): HeartRegionMaskState {
        val next = LinkedHashMap(policies)
        next[region.wireName] = policy
        val nextRevision = revision + 1
        return HeartRegionMaskState(
            revision = nextRevision,
            stateId = stateIdOf(nextRevision, next),
            policies = next,
        )
    }

    companion object {
        /** Defaults: all regions attended except training regions (none). */
        fun defaultPolicyFor(region: LogicalRegion): RegionMaskPolicy =
            if (region in TRAINING_REGIONS) RegionMaskPolicy.none() else RegionMaskPolicy.all()

        fun stateIdOf(revision: Long, policies: Map<String, RegionMaskPolicy>): String =
            canonicalSha256(
                mapOf(
                    "revision" to revision,
                    "policies" to policies.toSortedMap().mapValues { it.value.toCanonicalDict() },
                )
            )

        fun defaults(): HeartRegionMaskState {
            val policies = CANONICAL_REGION_ORDER.associate { it.wireName to defaultPolicyFor(it) }
            return HeartRegionMaskState(
                revision = 0,
                stateId = stateIdOf(0, policies),
                policies = policies,
            )
        }
    }
}
