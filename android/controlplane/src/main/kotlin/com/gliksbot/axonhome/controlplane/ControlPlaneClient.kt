package com.gliksbot.axonhome.controlplane

/**
 * Typed client contract for the Axon Control Plane v1 REST surface
 * (controlplane/openapi/axon-control-plane-v1.yaml).
 *
 * This is an INTERFACE ONLY — no network implementation lives in this module.
 * Implementations (Ktor/OkHttp/etc.) belong to :app; the deterministic
 * simulator in :core implements the same contract for offline/demo mode.
 *
 * Rules every implementation must honour (arch §5):
 * - TLS + bearer token (device enrolment); certificate errors never silently accepted.
 * - All mutating calls send an Idempotency-Key; retries never double-apply.
 * - Error body is ApiErrorEnvelope; unknown schema versions → 426.
 * - Governed failures pass through verbatim: trainer commands that return
 *   UNAVAILABLE surface as [TrainerCommandResponse] with
 *   status == UNAVAILABLE (HTTP 409), never as success.
 * - Masks use two-step confirm semantics; stop results use honest ack accounting.
 */
interface ControlPlaneClient {

    // ---- health ----

    /** GET /v1/health — liveness + distilled Heart health. */
    suspend fun health(): HealthResponse

    // ---- field (canonical; read-only) ----

    /** GET /v1/field/head — branch HEAD + health. */
    suspend fun fieldHead(): FieldHead

    /** GET /v1/field/snapshot/{fieldId} — snapshot reference view. */
    suspend fun fieldSnapshot(fieldId: String): FieldSnapshotView

    /** GET /v1/field/ticks/{n} — last n journal entries with delta refs. */
    suspend fun fieldTicks(count: Int): TickList

    // ---- masks (derived view; two-step confirm) ----

    /** GET /v1/masks — current mask state. */
    suspend fun masks(): MaskState

    /**
     * PUT /v1/masks. Pass confirmed=false to obtain a proposal + confirm_token;
     * pass the token with confirmed=true to apply. Never mutates on first touch.
     */
    suspend fun putMasks(request: PutMasksRequest, idempotencyKey: String): PutMasksResponse

    // ---- cores / souls ----

    /** GET /v1/cores. */
    suspend fun cores(): CoreList

    /** GET /v1/cores/{coreId}. */
    suspend fun core(coreId: String): CoreDetail

    /** GET /v1/souls/{coreId} — metadata only; payloads opaque. */
    suspend fun soul(coreId: String): SoulSummary

    // ---- trainer (fail-closed passthrough) ----

    /** GET /v1/trainer/status. */
    suspend fun trainerStatus(): TrainerStatus

    /**
     * POST /v1/trainer/commands. On UNAVAILABLE/REJECTED the server returns 409
     * with the full command result — implementations must surface that body,
     * not throw away the status.
     */
    suspend fun trainerCommand(request: TrainerCommandRequest): TrainerCommandResponse

    // ---- compute ----

    /** GET /v1/compute/workers. */
    suspend fun workers(): WorkerList

    /** POST /v1/compute/workers. */
    suspend fun registerWorker(request: RegisterWorkerRequest, idempotencyKey: String): Worker

    /** POST /v1/compute/workers/{workerId}/stop — honest ack; never fakes cancellation. */
    suspend fun stopWorker(workerId: String, idempotencyKey: String): WorkerStopResponse

    // ---- storage ----

    /** GET /v1/storage/artifacts. */
    suspend fun artifacts(): ArtifactList

    /** POST /v1/storage/artifacts. */
    suspend fun registerArtifact(request: RegisterArtifactRequest, idempotencyKey: String): Artifact

    // ---- capsules ----

    /** GET /v1/capsules. */
    suspend fun capsules(): CapsuleList

    /** POST /v1/capsules — export request; returns the manifest (written last). */
    suspend fun exportCapsule(request: CapsuleExportRequest, idempotencyKey: String): CapsuleManifest

    /** POST /v1/capsules/{capsuleId}/verify — fail-closed restore verification (arch §9.3). */
    suspend fun verifyCapsule(capsuleId: String, idempotencyKey: String): CapsuleVerificationReport

    // ---- audit ----

    /** GET /v1/audit — operational audit log (NOT the Engineer's Ledger). */
    suspend fun audit(limit: Int = 100, before: kotlinx.datetime.Instant? = null): AuditList

    // ---- agents ----

    /** GET /v1/agents. */
    suspend fun agents(): AgentList

    /** POST /v1/agents — vault key REFERENCE only, never raw keys. */
    suspend fun registerAgent(request: RegisterAgentRequest, idempotencyKey: String): AgentRecord

    /** POST /v1/agents/{agentId}/messages. */
    suspend fun postAgentMessage(agentId: String, request: AgentMessageRequest, idempotencyKey: String): AgentMessage

    // ---- stop (arch §13) ----

    /**
     * POST /v1/stop. The response's allStopped must be re-verified with
     * [StopAccounting.verify] before showing any "everything stopped" UI.
     */
    suspend fun stop(request: StopRequest, idempotencyKey: String): StopResponse

    // ---- events (WS primary / SSE fallback) ----

    /**
     * Open the typed event stream (axon-home-event-v1). Transport-specific:
     * implementations return a cold Flow that reconnects with [lastSeq] and
     * surfaces SEQ_GAP as ApiError(code = SEQ_GAP) so the caller re-syncs via
     * REST. Feed events through [SeqGapTracker] — a gap is never silently skipped.
     */
    fun events(lastSeq: Long? = null): kotlinx.coroutines.flow.Flow<AxonEvent>
}
