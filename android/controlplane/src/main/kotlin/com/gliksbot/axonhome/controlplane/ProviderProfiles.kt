package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/** Provider identity is intentionally vendor-neutral. */
@Serializable
enum class ProviderTransport {
    @SerialName("local") LOCAL,
    @SerialName("http") HTTP,
    @SerialName("https") HTTPS,
    @SerialName("ssh") SSH,
}

@Serializable
enum class ProviderPurpose {
    @SerialName("reasoning") REASONING,
    @SerialName("training") TRAINING,
}

@Serializable
enum class ProviderAuthKind {
    @SerialName("none") NONE,
    @SerialName("bearer") BEARER,
    @SerialName("api_key") API_KEY,
    @SerialName("ssh_key") SSH_KEY,
    @SerialName("password") PASSWORD,
}

/**
 * Saved operator profile. SecretRef names Android-Keystore-backed secret
 * material; it is never the secret itself.
 */
@Serializable
data class ProviderProfile(
    @SerialName("provider_id") val providerId: String,
    val label: String,
    val transport: ProviderTransport,
    val endpoint: String,
    val purposes: Set<ProviderPurpose>,
    @SerialName("auth_kind") val authKind: ProviderAuthKind = ProviderAuthKind.NONE,
    @SerialName("secret_ref") val secretRef: String? = null,
    val username: String? = null,
    val capabilities: CapabilitySet = CapabilitySet(),
    val accelerator: String? = null,
    @SerialName("created_at") val createdAt: Instant,
    @SerialName("last_seen_at") val lastSeenAt: Instant? = null,
)

/** Anatomy shown for every selectable Core/checkpoint. */
@Serializable
data class CoreAnatomyView(
    @SerialName("d_model") val dModel: Int,
    @SerialName("n_heads") val nHeads: Int,
    @SerialName("n_layers") val nLayers: Int,
    @SerialName("ffn_dim") val ffnDim: Int,
    @SerialName("state_tokens") val stateTokens: Int? = null,
    @SerialName("page_size") val pageSize: Int? = null,
    @SerialName("parameter_count") val parameterCount: Long? = null,
    @SerialName("parameter_bytes") val parameterBytes: Long? = null,
)

/** Operator-facing continuation/checkpoint summary. */
@Serializable
data class TrainingCheckpointView(
    @SerialName("checkpoint_id") val checkpointId: String,
    @SerialName("core_id") val coreId: String,
    @SerialName("candidate_generation_id") val candidateGenerationId: String,
    @SerialName("architecture_id") val architectureId: String,
    val anatomy: CoreAnatomyView,
    @SerialName("optimizer_step") val optimizerStep: Long,
    val phase: String? = null,
    val stage: String? = null,
    @SerialName("curriculum_id") val curriculumId: String? = null,
    @SerialName("curriculum_name") val curriculumName: String? = null,
    @SerialName("latest_loss") val latestLoss: Double? = null,
    @SerialName("accepted") val accepted: Boolean = false,
    @SerialName("recovery_slot") val recoverySlot: String? = null,
    @SerialName("artifact_sha256") val artifactSha256: String? = null,
    @SerialName("artifact_bytes") val artifactBytes: Long? = null,
    @SerialName("created_at") val createdAt: Instant? = null,
)

@Serializable
data class CurriculumSummary(
    @SerialName("curriculum_id") val curriculumId: String,
    val name: String,
    val stage: String? = null,
    val phase: String? = null,
    val description: String? = null,
    @SerialName("lesson_count") val lessonCount: Long? = null,
    @SerialName("sealed") val sealed: Boolean = true,
)

/**
 * Editable draft. `spec` is provider-independent curriculum content. Saving a
 * modified sealed curriculum must create a new identity rather than mutate the
 * prior one.
 */
@Serializable
data class CurriculumDraft(
    val name: String,
    val stage: String? = null,
    val phase: String? = null,
    val description: String? = null,
    val spec: JsonObject,
)

@Serializable
enum class TrainingSessionStatus {
    @SerialName("prepared") PREPARED,
    @SerialName("running") RUNNING,
    @SerialName("boundary_sync") BOUNDARY_SYNC,
    @SerialName("paused") PAUSED,
    @SerialName("completed") COMPLETED,
    @SerialName("failed") FAILED,
    @SerialName("lost") LOST,
}

/** Live sample-level observation for the Trainer monitor. */
@Serializable
data class TrainingObservation(
    @SerialName("session_id") val sessionId: String,
    @SerialName("observed_at") val observedAt: Instant,
    @SerialName("optimizer_step") val optimizerStep: Long,
    @SerialName("tranche_step") val trancheStep: Int,
    @SerialName("tranche_size") val trancheSize: Int,
    val loss: Double? = null,
    @SerialName("learning_rate") val learningRate: Double? = null,
    val input: String? = null,
    val target: String? = null,
    val prediction: String? = null,
    val exact: Boolean? = null,
    val gates: Map<String, Double> = emptyMap(),
    val details: JsonObject = JsonObject(emptyMap()),
)

@Serializable
data class RecoverySlotState(
    val slot: String,
    @SerialName("checkpoint_id") val checkpointId: String? = null,
    @SerialName("optimizer_step") val optimizerStep: Long? = null,
    val status: String,
    @SerialName("verified_at") val verifiedAt: Instant? = null,
)

@Serializable
data class TrainingSessionView(
    @SerialName("session_id") val sessionId: String,
    val status: TrainingSessionStatus,
    @SerialName("provider_id") val providerId: String,
    @SerialName("worker_id") val workerId: String? = null,
    @SerialName("core_id") val coreId: String,
    @SerialName("checkpoint_id") val checkpointId: String,
    @SerialName("curriculum_id") val curriculumId: String,
    val anatomy: CoreAnatomyView,
    @SerialName("optimizer_step") val optimizerStep: Long,
    @SerialName("tranche_size") val trancheSize: Int = 15,
    @SerialName("tranche_step") val trancheStep: Int = 0,
    val phase: String? = null,
    val stage: String? = null,
    val loss: Double? = null,
    @SerialName("recovery_slots") val recoverySlots: List<RecoverySlotState> = emptyList(),
    @SerialName("last_locally_verified_step") val lastLocallyVerifiedStep: Long? = null,
    @SerialName("latest_observation") val latestObservation: TrainingObservation? = null,
)

@Serializable
data class TrainingWorkspaceSnapshot(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val providers: List<ProviderProfile>,
    val checkpoints: List<TrainingCheckpointView>,
    val curricula: List<CurriculumSummary>,
    @SerialName("active_sessions") val activeSessions: List<TrainingSessionView>,
) {
    companion object {
        const val SCHEMA = "axon-training-workspace-v1"
    }
}
