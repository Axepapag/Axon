package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/** Exactly the 13 TrainerCommandKind values from runtime/trainer/organ.py. */
@Serializable
enum class TrainerCommandKind {
    @SerialName("status") STATUS,
    @SerialName("inventory") INVENTORY,
    @SerialName("preflight") PREFLIGHT,
    @SerialName("configure") CONFIGURE,
    @SerialName("start") START,
    @SerialName("pause") PAUSE,
    @SerialName("resume") RESUME,
    @SerialName("evaluate") EVALUATE,
    @SerialName("export_cloud_packet") EXPORT_CLOUD_PACKET,
    @SerialName("import_cloud_result") IMPORT_CLOUD_RESULT,
    @SerialName("compare") COMPARE,
    @SerialName("request_promotion") REQUEST_PROMOTION,
    @SerialName("rollback") ROLLBACK,
}

/**
 * TrainerCommandStatus. The organ is FAIL-CLOSED: unwired kinds return
 * UNAVAILABLE ("no governed handler; no action was taken") and the API returns
 * 409 with the status verbatim. The app renders UNAVAILABLE as
 * "Not implemented", never success (arch §5.5).
 */
@Serializable
enum class TrainerCommandStatus {
    OK,
    UNAVAILABLE,
    REJECTED,
    FAILED,
}

/** GET /v1/trainer/status. */
@Serializable
data class TrainerStatus(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val inspection: TrainerInspection,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-trainer-status-v1"
    }
}

/** Distilled inspect_trainer_state view (full record schema axon-trainer-inspection-v1). */
@Serializable
data class TrainerInspection(
    val schema: String = SCHEMA,
    /** TrainerWriterLease at training/trainer/authority/lease.json. */
    @SerialName("writer_lease_present") val writerLeasePresent: Boolean,
    @SerialName("lease_owner") val leaseOwner: String? = null,
    @SerialName("active_generations") val activeGenerations: List<ActiveGeneration>,
    @SerialName("latest_checkpoints") val latestCheckpoints: List<CheckpointRecordMeta>,
    /**
     * Per-TrainerCommandKind availability map mirroring the fail-closed organ
     * registry. Only kinds with a wired governed handler are true.
     */
    @SerialName("available_commands") val availableCommands: Map<TrainerCommandKind, Boolean> = emptyMap(),
) {
    companion object {
        const val SCHEMA = "axon-trainer-inspection-v1"
    }
}

@Serializable
enum class TrainerLifecycleStatus {
    @SerialName("prepared") PREPARED,
    @SerialName("running") RUNNING,
    @SerialName("paused") PAUSED,
    @SerialName("completed") COMPLETED,
    @SerialName("rejected") REJECTED,
    @SerialName("gate_passed") GATE_PASSED,
    @SerialName("promotion_proposed") PROMOTION_PROPOSED,
    @SerialName("promoted") PROMOTED,
    @SerialName("retired") RETIRED,
}

@Serializable
data class ActiveGeneration(
    @SerialName("module_id") val moduleId: String,
    @SerialName("candidate_generation_id") val candidateGenerationId: String,
    @SerialName("base_generation_id") val baseGenerationId: String? = null,
    @SerialName("lifecycle_status") val lifecycleStatus: TrainerLifecycleStatus,
    @SerialName("current_step") val currentStep: Long? = null,
    @SerialName("latest_loss") val latestLoss: Double? = null,
    @SerialName("learning_rate") val learningRate: Double? = null,
)

/** Checkpoint RECORD summary (axon-trainer-candidate-checkpoint-v2 record, not payload). */
@Serializable
data class CheckpointRecordMeta(
    val schema: String = SCHEMA,
    @SerialName("checkpoint_id") val checkpointId: String,
    @SerialName("module_id") val moduleId: String,
    @SerialName("candidate_generation_id") val candidateGenerationId: String,
    val step: Long,
    @SerialName("artifact_relpath") val artifactRelpath: String? = null,
    @SerialName("artifact_sha256") val artifactSha256: String,
    @SerialName("artifact_bytes") val artifactBytes: Long? = null,
    @SerialName("optimizer_included") val optimizerIncluded: Boolean = false,
) {
    companion object {
        const val SCHEMA = "axon-trainer-candidate-checkpoint-v2"
    }
}

/** POST /v1/trainer/commands request. */
@Serializable
data class TrainerCommandRequest(
    val schema: String = SCHEMA,
    val kind: TrainerCommandKind,
    /** Kind-specific arguments, passed verbatim to TrainerOrganCommand.args. */
    val args: JsonObject = JsonObject(emptyMap()),
    /** Authenticated operator identity (audit "who"). */
    @SerialName("requested_by") val requestedBy: String,
    /** Client-generated; retries never double-apply. */
    @SerialName("idempotency_key") val idempotencyKey: String,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-trainer-command-v1"
    }
}

/** POST /v1/trainer/commands response (200 OK; also the 409 body for UNAVAILABLE/REJECTED passthrough). */
@Serializable
data class TrainerCommandResponse(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    @SerialName("command_id") val commandId: String,
    val kind: TrainerCommandKind,
    val status: TrainerCommandStatus,
    /** Organ message verbatim, e.g. "no governed handler; no action was taken". */
    val message: String,
    /** Kind-specific result payload (e.g. axon-trainer-organ-status-summary-v1 for kind=status). */
    val result: JsonObject? = null,
    /** Audit entry recording this dispatch (arch §14). */
    @SerialName("audit_id") val auditId: String? = null,
    /** True when this is the recorded result of an earlier Idempotency-Key. */
    @SerialName("idempotent_replay") val idempotentReplay: Boolean = false,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-trainer-command-result-v1"
    }
}
