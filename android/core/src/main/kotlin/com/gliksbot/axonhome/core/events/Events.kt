package com.gliksbot.axonhome.core.events

import com.gliksbot.axonhome.core.domain.canonicalSha256
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Typed Axon event stream v1 (mission §7). Events are versioned by schema
 * and replayable by sequence number. Payloads are polymorphic on "type";
 * the envelope [AxonEvent.type] duplicates the payload type for cheap
 * filtering without deserializing the payload.
 */

const val AXON_EVENT_SCHEMA = "axon-home-event-v1"

@Serializable
sealed class AxonEventPayload {
    abstract val eventType: String

    @Serializable
    @SerialName("TickStarted")
    data class TickStarted(
        @SerialName("tick_id") val tickId: Int,
        @SerialName("base_field_id") val baseFieldId: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "TickStarted"
    }

    @Serializable
    @SerialName("CoreFirstReturned")
    data class CoreFirstReturned(
        @SerialName("tick_id") val tickId: Int,
        @SerialName("core_id") val coreId: String,
        @SerialName("proposal_id") val proposalId: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "CoreFirstReturned"
    }

    @Serializable
    @SerialName("FirstBarrierComplete")
    data class FirstBarrierComplete(
        @SerialName("tick_id") val tickId: Int,
        val participants: List<String>,
    ) : AxonEventPayload() {
        override val eventType: String get() = "FirstBarrierComplete"
    }

    @Serializable
    @SerialName("CoreRefinedReturned")
    data class CoreRefinedReturned(
        @SerialName("tick_id") val tickId: Int,
        @SerialName("core_id") val coreId: String,
        @SerialName("proposal_id") val proposalId: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "CoreRefinedReturned"
    }

    @Serializable
    @SerialName("RefinedBarrierComplete")
    data class RefinedBarrierComplete(
        @SerialName("tick_id") val tickId: Int,
        val participants: List<String>,
    ) : AxonEventPayload() {
        override val eventType: String get() = "RefinedBarrierComplete"
    }

    @Serializable
    @SerialName("ConsolidatorSelected")
    data class ConsolidatorSelected(
        @SerialName("tick_id") val tickId: Int,
        @SerialName("core_id") val coreId: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "ConsolidatorSelected"
    }

    @Serializable
    @SerialName("FinalReturned")
    data class FinalReturned(
        @SerialName("tick_id") val tickId: Int,
        @SerialName("core_id") val coreId: String,
        @SerialName("verdict_id") val verdictId: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "FinalReturned"
    }

    @Serializable
    @SerialName("HeartValidationStarted")
    data class HeartValidationStarted(
        @SerialName("tick_id") val tickId: Int,
        @SerialName("verdict_id") val verdictId: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "HeartValidationStarted"
    }

    @Serializable
    @SerialName("FieldCommitted")
    data class FieldCommitted(
        @SerialName("tick_id") val tickId: Int,
        @SerialName("field_id") val fieldId: String,
        @SerialName("parent_field_id") val parentFieldId: String?,
        @SerialName("delta_ids") val deltaIds: List<String>,
    ) : AxonEventPayload() {
        override val eventType: String get() = "FieldCommitted"
    }

    @Serializable
    @SerialName("SoulTransitionAccepted")
    data class SoulTransitionAccepted(
        @SerialName("core_id") val coreId: String,
        @SerialName("soul_id") val soulId: String,
        val generation: Int,
        @SerialName("receipt_id") val receiptId: String,
        val phase: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "SoulTransitionAccepted"
    }

    @Serializable
    @SerialName("TrainerStepAccepted")
    data class TrainerStepAccepted(
        @SerialName("module_id") val moduleId: String,
        @SerialName("candidate_generation_id") val candidateGenerationId: String,
        val step: Int,
        val loss: Double,
        @SerialName("gradient_l2") val gradientL2: Double,
        @SerialName("bundle_id") val bundleId: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "TrainerStepAccepted"
    }

    @Serializable
    @SerialName("CheckpointWritten")
    data class CheckpointWritten(
        @SerialName("checkpoint_id") val checkpointId: String,
        val step: Int,
        @SerialName("candidate_generation_id") val candidateGenerationId: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "CheckpointWritten"
    }

    @Serializable
    @SerialName("HeldoutEvaluationCompleted")
    data class HeldoutEvaluationCompleted(
        @SerialName("evaluation_id") val evaluationId: String,
        @SerialName("candidate_generation_id") val candidateGenerationId: String,
        val metrics: Map<String, Double>,
    ) : AxonEventPayload() {
        override val eventType: String get() = "HeldoutEvaluationCompleted"
    }

    @Serializable
    @SerialName("ComputeWorkerConnected")
    data class ComputeWorkerConnected(
        @SerialName("worker_id") val workerId: String,
        val provider: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "ComputeWorkerConnected"
    }

    @Serializable
    @SerialName("ComputeWorkerLost")
    data class ComputeWorkerLost(
        @SerialName("worker_id") val workerId: String,
        val reason: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "ComputeWorkerLost"
    }

    @Serializable
    @SerialName("RecoveryCapsulePublished")
    data class RecoveryCapsulePublished(
        @SerialName("capsule_id") val capsuleId: String,
        @SerialName("manifest_sha256") val manifestSha256: String,
        val members: List<String>,
    ) : AxonEventPayload() {
        override val eventType: String get() = "RecoveryCapsulePublished"
    }

    @Serializable
    @SerialName("AgentMessageReceived")
    data class AgentMessageReceived(
        @SerialName("agent_id") val agentId: String,
        val channel: String,
        val text: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "AgentMessageReceived"
    }

    @Serializable
    @SerialName("AlertRaised")
    data class AlertRaised(
        val severity: String,
        val message: String,
        val source: String,
    ) : AxonEventPayload() {
        override val eventType: String get() = "AlertRaised"
    }
}

@Serializable
data class AxonEvent(
    @SerialName("event_id") val eventId: String,
    val seq: Long,
    val timestamp: String,
    val type: String,
    val payload: AxonEventPayload,
    @SerialName("schema") val schema: String = AXON_EVENT_SCHEMA,
) {
    init {
        require(schema == AXON_EVENT_SCHEMA) { "unsupported event schema '$schema'" }
        require(seq >= 0) { "event seq must be non-negative" }
    }
}

/**
 * In-memory event log with monotonic sequence assignment and replay.
 * Sequence 0 is the first event. Replay is read-only evidence; it never
 * mutates live state.
 */
class EventLog {
    private val _events = mutableListOf<AxonEvent>()

    val events: List<AxonEvent> get() = _events.toList()
    val nextSeq: Long get() = _events.size.toLong()

    fun append(payload: AxonEventPayload, timestamp: String): AxonEvent {
        val seq = nextSeq
        val event = AxonEvent(
            eventId = "evt-" + canonicalSha256("$seq|${payload.eventType}|$timestamp").take(24),
            seq = seq,
            timestamp = timestamp,
            type = payload.eventType,
            payload = payload,
        )
        _events.add(event)
        return event
    }

    /** All events with seq >= [fromSeq], in order. */
    fun replayFrom(fromSeq: Long): List<AxonEvent> =
        _events.filter { it.seq >= fromSeq }

    fun ofType(type: String): List<AxonEvent> = _events.filter { it.type == type }
}
