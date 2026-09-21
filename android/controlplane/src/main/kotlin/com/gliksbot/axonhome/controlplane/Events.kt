package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Typed event stream (arch §6). Envelope `axon-home-event-v1`:
 * `{schema, event_id, seq, timestamp, type, payload}`. `seq` is a monotonic
 * per-stream sequence; reconnect with last_seq; if older than retention the
 * server responds SEQ_GAP and the client re-syncs via REST. A visible seq gap
 * is never silently skipped.
 *
 * Decoding: [AxonEvent.json] uses `type` as the class discriminator, so
 * `AxonEvent.json.decodeFromString<AxonEvent>(text)` yields the typed subclass
 * with its typed payload.
 */
@Serializable
sealed class AxonEvent {
    abstract val schema: String
    abstract val eventId: String
    abstract val seq: Long
    abstract val timestamp: Instant

    companion object {
        const val ENVELOPE_SCHEMA = "axon-home-event-v1"

        /** Json configured for the event envelope (type discriminator = "type"). */
        val json: Json = Json {
            classDiscriminator = "type"
            ignoreUnknownKeys = true
        }
    }

    /** Event type name as it appears on the wire (envelope "type" field). */
    val typeName: String get() = this::class.simpleName ?: "Unknown"

    /** #1 — payload refs axon-heart-tick-identity-v1. */
    @Serializable
    @SerialName("TickStarted")
    data class TickStarted(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: TickStartedPayload,
    ) : AxonEvent()

    /** #2 — payload refs axon-english-proposal-v1 + participant record. */
    @Serializable
    @SerialName("CoreFirstReturned")
    data class CoreFirstReturned(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: ProposalReturnedPayload,
    ) : AxonEvent()

    /** #3 — payload refs axon-heart-english-proposal-workspace-v2 (first pass). */
    @Serializable
    @SerialName("FirstBarrierComplete")
    data class FirstBarrierComplete(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: BarrierCompletePayload,
    ) : AxonEvent()

    /** #4 — payload refs axon-english-proposal-v1 (refined pass). */
    @Serializable
    @SerialName("CoreRefinedReturned")
    data class CoreRefinedReturned(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: ProposalReturnedPayload,
    ) : AxonEvent()

    /** #5 — payload refs axon-heart-english-proposal-workspace-v2 (refined pass). */
    @Serializable
    @SerialName("RefinedBarrierComplete")
    data class RefinedBarrierComplete(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: BarrierCompletePayload,
    ) : AxonEvent()

    /** #6 — consolidator = participants[(tick_sequence - 1) % len(participants)]. */
    @Serializable
    @SerialName("ConsolidatorSelected")
    data class ConsolidatorSelected(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: ConsolidatorSelectedPayload,
    ) : AxonEvent()

    /** #7 — payload refs axon-tagged-final-verdict-v2. */
    @Serializable
    @SerialName("FinalReturned")
    data class FinalReturned(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: FinalReturnedPayload,
    ) : AxonEvent()

    /** #8 — payload refs axon-heart-frozen-tick-image-v2. */
    @Serializable
    @SerialName("HeartValidationStarted")
    data class HeartValidationStarted(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: HeartValidationStartedPayload,
    ) : AxonEvent()

    /** #9 — payload refs axon-heart-commit-v1 + axon-canonical-state-branch-event-v1. */
    @Serializable
    @SerialName("FieldCommitted")
    data class FieldCommitted(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: FieldCommittedPayload,
    ) : AxonEvent()

    /** #10 — payload refs axon-private-soul-commit-receipt-v1. */
    @Serializable
    @SerialName("SoulTransitionAccepted")
    data class SoulTransitionAccepted(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: SoulTransitionAcceptedPayload,
    ) : AxonEvent()

    /** #11 — payload refs axon-accepted-reasoning-training-step-v1 + pointer. */
    @Serializable
    @SerialName("TrainerStepAccepted")
    data class TrainerStepAccepted(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: TrainerStepAcceptedPayload,
    ) : AxonEvent()

    /** #12 — payload refs the axon-trainer-candidate-checkpoint-v2 RECORD (not payload). */
    @Serializable
    @SerialName("CheckpointWritten")
    data class CheckpointWritten(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: CheckpointWrittenPayload,
    ) : AxonEvent()

    /** #13 — evaluation record id + gate requirements result. */
    @Serializable
    @SerialName("HeldoutEvaluationCompleted")
    data class HeldoutEvaluationCompleted(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: HeldoutEvaluationCompletedPayload,
    ) : AxonEvent()

    /** #14 — worker registry record + capabilities. */
    @Serializable
    @SerialName("ComputeWorkerConnected")
    data class ComputeWorkerConnected(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: ComputeWorkerConnectedPayload,
    ) : AxonEvent()

    /** #15 — worker id + last heartbeat + active job. */
    @Serializable
    @SerialName("ComputeWorkerLost")
    data class ComputeWorkerLost(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: ComputeWorkerLostPayload,
    ) : AxonEvent()

    /** #16 — capsule manifest (axon-cloud-bundle-manifest-v1-compatible). */
    @Serializable
    @SerialName("RecoveryCapsulePublished")
    data class RecoveryCapsulePublished(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: RecoveryCapsulePublishedPayload,
    ) : AxonEvent()

    /** #17 — Roundtable message ref; NEVER carries secrets. */
    @Serializable
    @SerialName("AgentMessageReceived")
    data class AgentMessageReceived(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: AgentMessageReceivedPayload,
    ) : AxonEvent()

    /** #18 — severity, source, message, related event ids. */
    @Serializable
    @SerialName("AlertRaised")
    data class AlertRaised(
        override val schema: String = ENVELOPE_SCHEMA,
        @SerialName("event_id") override val eventId: String,
        override val seq: Long,
        override val timestamp: Instant,
        val payload: AlertRaisedPayload,
    ) : AxonEvent()
}

@Serializable
data class TickStartedPayload(
    val schema: String = "axon-heart-tick-identity-v1",
    @SerialName("tick_sequence") val tickSequence: Long,
    @SerialName("heartbeat_id") val heartbeatId: String,
    @SerialName("base_field_id") val baseFieldId: String,
    @SerialName("base_tick_id") val baseTickId: Long? = null,
    @SerialName("view_id") val viewId: String? = null,
    val participants: List<String> = emptyList(),
)

@Serializable
enum class ParticipantState {
    @SerialName("pending") PENDING,
    @SerialName("returned") RETURNED,
    @SerialName("failed") FAILED,
    @SerialName("timed_out") TIMED_OUT,
}

@Serializable
data class ProposalReturnedPayload(
    val schema: String = "axon-english-proposal-v1",
    @SerialName("tick_sequence") val tickSequence: Long,
    @SerialName("core_id") val coreId: String,
    @SerialName("proposal_id") val proposalId: String,
    @SerialName("participant_state") val participantState: ParticipantState,
    @SerialName("elapsed_ms") val elapsedMs: Long? = null,
)

@Serializable
data class BarrierCompletePayload(
    val schema: String = "axon-heart-english-proposal-workspace-v2",
    @SerialName("tick_sequence") val tickSequence: Long,
    @SerialName("workspace_id") val workspaceId: String,
    val pass: String? = null,
    @SerialName("returned_core_ids") val returnedCoreIds: List<String>,
    @SerialName("failed_core_ids") val failedCoreIds: List<String> = emptyList(),
)

@Serializable
data class ConsolidatorSelectedPayload(
    @SerialName("tick_sequence") val tickSequence: Long,
    @SerialName("core_id") val coreId: String,
    @SerialName("participant_count") val participantCount: Int,
)

@Serializable
data class FinalReturnedPayload(
    val schema: String = "axon-tagged-final-verdict-v2",
    @SerialName("tick_sequence") val tickSequence: Long,
    @SerialName("consolidator_core_id") val consolidatorCoreId: String,
    @SerialName("verdict_id") val verdictId: String,
    /** e.g. #responseDraft#, #scratch#, #journal#. */
    @SerialName("region_tags") val regionTags: List<String> = emptyList(),
)

@Serializable
data class HeartValidationStartedPayload(
    val schema: String = "axon-heart-frozen-tick-image-v2",
    @SerialName("tick_sequence") val tickSequence: Long,
    @SerialName("frozen_image_id") val frozenImageId: String,
    @SerialName("view_id") val viewId: String,
)

@Serializable
data class FieldCommittedPayload(
    val schema: String = "axon-heart-commit-v1",
    @SerialName("tick_sequence") val tickSequence: Long,
    @SerialName("base_field_id") val baseFieldId: String,
    @SerialName("successor_field_id") val successorFieldId: String,
    /** shared-field-delta-v1 ids. */
    @SerialName("delta_ids") val deltaIds: List<String>,
    @SerialName("commit_id") val commitId: String,
    /** Generation in the branch journal (axon-canonical-state-branch-event-v1). */
    @SerialName("journal_generation") val journalGeneration: Long? = null,
)

@Serializable
data class SoulTransitionAcceptedPayload(
    val schema: String = "axon-private-soul-commit-receipt-v1",
    @SerialName("core_id") val coreId: String,
    @SerialName("receipt_id") val receiptId: String,
    @SerialName("transition_id") val transitionId: String? = null,
    @SerialName("before_soul_id") val beforeSoulId: String,
    @SerialName("after_soul_id") val afterSoulId: String,
    val generation: Long,
    val phase: SoulPhase,
    @SerialName("tick_uid") val tickUid: String? = null,
)

@Serializable
data class TrainerStepAcceptedPayload(
    val schema: String = "axon-accepted-reasoning-training-step-v1",
    @SerialName("bundle_id") val bundleId: String,
    /** axon-accepted-training-step-pointer-v1 id. */
    @SerialName("pointer_id") val pointerId: String,
    @SerialName("module_id") val moduleId: String,
    @SerialName("candidate_generation_id") val candidateGenerationId: String,
    @SerialName("core_id") val coreId: String? = null,
    val step: Long,
    @SerialName("checkpoint_id") val checkpointId: String? = null,
    @SerialName("before_soul_id") val beforeSoulId: String? = null,
    @SerialName("after_soul_id") val afterSoulId: String? = null,
    @SerialName("previous_bundle_id") val previousBundleId: String? = null,
)

@Serializable
data class CheckpointWrittenPayload(
    val schema: String = "axon-trainer-candidate-checkpoint-v2",
    @SerialName("checkpoint_id") val checkpointId: String,
    @SerialName("module_id") val moduleId: String,
    @SerialName("candidate_generation_id") val candidateGenerationId: String,
    val step: Long,
    @SerialName("artifact_relpath") val artifactRelpath: String? = null,
    @SerialName("artifact_sha256") val artifactSha256: String,
    @SerialName("artifact_bytes") val artifactBytes: Long? = null,
    @SerialName("optimizer_included") val optimizerIncluded: Boolean = false,
)

@Serializable
data class HeldoutEvaluationCompletedPayload(
    @SerialName("evaluation_id") val evaluationId: String,
    @SerialName("candidate_generation_id") val candidateGenerationId: String,
    @SerialName("module_id") val moduleId: String? = null,
    @SerialName("gate_passed") val gatePassed: Boolean,
    val metrics: Map<String, Double> = emptyMap(),
    @SerialName("failed_requirements") val failedRequirements: List<String> = emptyList(),
)

@Serializable
data class ComputeWorkerConnectedPayload(
    @SerialName("worker_id") val workerId: String,
    val provider: String,
    val endpoint: String? = null,
    val capabilities: Map<String, Boolean> = emptyMap(),
    @SerialName("worker_version") val workerVersion: String? = null,
)

@Serializable
data class ComputeWorkerLostPayload(
    @SerialName("worker_id") val workerId: String,
    val provider: String? = null,
    @SerialName("last_heartbeat_at") val lastHeartbeatAt: Instant? = null,
    @SerialName("active_job_id") val activeJobId: String? = null,
)

@Serializable
data class RecoveryCapsulePublishedPayload(
    val schema: String = "axon-cloud-bundle-manifest-v1",
    @SerialName("capsule_id") val capsuleId: String,
    val kind: CapsuleKind? = null,
    @SerialName("archive_sha256") val archiveSha256: String,
    @SerialName("archive_bytes") val archiveBytes: Long,
    @SerialName("member_count") val memberCount: Int,
    @SerialName("git_commit") val gitCommit: String? = null,
)

@Serializable
data class AgentMessageReceivedPayload(
    @SerialName("message_id") val messageId: String,
    @SerialName("agent_id") val agentId: String,
    val from: String,
    val preview: String? = null,
    @SerialName("in_reply_to") val inReplyTo: String? = null,
)

@Serializable
enum class AlertSeverity {
    @SerialName("info") INFO,
    @SerialName("warning") WARNING,
    @SerialName("critical") CRITICAL,
}

@Serializable
data class AlertRaisedPayload(
    val severity: AlertSeverity,
    val source: String,
    val message: String,
    @SerialName("related_event_ids") val relatedEventIds: List<String> = emptyList(),
)

/**
 * Seq-gap detection for the event stream (arch §6.1). The client tracks the
 * last contiguous seq; a forward jump is a gap (re-sync via REST before
 * resuming); replayed/out-of-order events with seq <= last are dropped. A
 * visible gap is never silently skipped.
 */
class SeqGapTracker {
    /** Highest contiguous seq observed; null before the first event. */
    var lastSeq: Long? = null
        private set

    sealed class Result {
        /** In-order (or first) event; process it. */
        data object InOrder : Result()
        /** seq jumped forward: [missed] events (from..to inclusive) were not seen. Re-sync via REST. */
        data class Gap(val from: Long, val to: Long) : Result() {
            val missed: Long get() = to - from + 1
        }
        /** Replayed or out-of-order event (seq <= lastSeq); do not process. */
        data object Duplicate : Result()
    }

    /** Record an observed event seq. */
    fun observe(seq: Long): Result {
        val last = lastSeq
        return when {
            last == null -> {
                lastSeq = seq
                Result.InOrder
            }
            seq <= last -> Result.Duplicate
            seq == last + 1 -> {
                lastSeq = seq
                Result.InOrder
            }
            else -> {
                lastSeq = seq
                Result.Gap(from = last + 1, to = seq - 1)
            }
        }
    }

    /** Reset after a REST re-sync; the next event is treated as the first. */
    fun reset() {
        lastSeq = null
    }
}
