package com.gliksbot.axonhome.controlplane

import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/**
 * Provider-neutral Trainer workspace. This is intentionally separate from the
 * legacy/vendor launch surfaces: the phone-authoritative Trainer selects a
 * generic provider profile and a worker executes only the assigned work.
 */
interface TrainingWorkspaceClient {
    suspend fun trainingWorkspace(): TrainingWorkspaceSnapshot

    suspend fun saveProvider(request: SaveProviderProfileRequest, idempotencyKey: String): ProviderProfile
    suspend fun removeProvider(providerId: String, idempotencyKey: String)
    suspend fun probeProvider(providerId: String): ProviderProbeResult

    suspend fun createCore(request: CreateCoreRequest, idempotencyKey: String): TrainingCheckpointView
    suspend fun checkpoints(coreId: String? = null): List<TrainingCheckpointView>

    suspend fun curricula(): List<CurriculumSummary>
    suspend fun sealCurriculum(request: CurriculumDraft, idempotencyKey: String): CurriculumSummary

    suspend fun startTraining(request: StartTrainingRequest, idempotencyKey: String): TrainingSessionView
    suspend fun pauseTraining(sessionId: String, idempotencyKey: String): TrainingSessionView
    suspend fun resumeTraining(sessionId: String, idempotencyKey: String): TrainingSessionView
    suspend fun stopTraining(sessionId: String, idempotencyKey: String): TrainingSessionView
    suspend fun trainingSession(sessionId: String): TrainingSessionView
    fun trainingObservations(sessionId: String, lastSeq: Long? = null): Flow<TrainingObservation>
}

@Serializable
data class SaveProviderProfileRequest(
    val label: String,
    val transport: ProviderTransport,
    val endpoint: String,
    val purposes: Set<ProviderPurpose>,
    @SerialName("auth_kind") val authKind: ProviderAuthKind = ProviderAuthKind.NONE,
    @SerialName("secret_ref") val secretRef: String? = null,
    val username: String? = null,
    val options: JsonObject = JsonObject(emptyMap()),
)

@Serializable
data class ProviderProbeResult(
    @SerialName("provider_id") val providerId: String,
    val reachable: Boolean,
    val capabilities: CapabilitySet = CapabilitySet(),
    val accelerator: String? = null,
    @SerialName("worker_version") val workerVersion: String? = null,
    val detail: String? = null,
)

@Serializable
data class CreateCoreRequest(
    @SerialName("core_id") val coreId: String,
    val anatomy: CoreAnatomyView,
    /** Optional architecture family. Current implementation uses living-english. */
    @SerialName("architecture_family") val architectureFamily: String = "living-english",
    /** Create a fresh optimizer/Soul/checkpoint lineage. */
    @SerialName("genesis") val genesis: Boolean = true,
    val options: JsonObject = JsonObject(emptyMap()),
)

@Serializable
data class StartTrainingRequest(
    @SerialName("provider_id") val providerId: String,
    @SerialName("core_id") val coreId: String,
    @SerialName("checkpoint_id") val checkpointId: String,
    @SerialName("curriculum_id") val curriculumId: String,
    /** Recovery boundary. Default 15, operator-adjustable. */
    @SerialName("tranche_size") val trancheSize: Int = 15,
    /** Optional total target. Null means continue until paused/stopped/gate policy ends it. */
    @SerialName("target_optimizer_step") val targetOptimizerStep: Long? = null,
    val options: JsonObject = JsonObject(emptyMap()),
)
