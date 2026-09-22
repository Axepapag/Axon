package com.gliksbot.axonhome.sim

import com.gliksbot.axonhome.controlplane.*
import com.gliksbot.axonhome.core.capsule.RecoveryCapsule
import com.gliksbot.axonhome.core.capsule.RecoveryCapsuleVerifier
import com.gliksbot.axonhome.core.domain.CORE_WRITABLE_REGIONS
import com.gliksbot.axonhome.core.domain.LogicalRegion as CoreRegion
import com.gliksbot.axonhome.core.domain.RegionMaskPolicy
import com.gliksbot.axonhome.core.domain.SharedFieldSnapshot
import com.gliksbot.axonhome.core.domain.canonicalSha256
import com.gliksbot.axonhome.core.events.AxonEventPayload
import com.gliksbot.axonhome.core.sim.SIMULATION_MARKER
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.core.stop.StopKind
import com.gliksbot.axonhome.core.stop.StopResult
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.datetime.Instant

/**
 * Simulated Control Plane: adapts :core's deterministic [SimulationEngine]
 * to the production [ControlPlaneClient] contract so the app consumes the
 * simulator through exactly the same typed surface it will use against the
 * real Control Plane (arch §11). No network. Every response is freshness
 * LIVE but carries SIMULATION provenance via the app's banner layer; payloads
 * themselves embed the SIMULATION marker where the domain provides one.
 */
class SimControlPlaneClient(
    val engine: SimulationEngine,
) : ControlPlaneClient {

    private var clockSeconds = com.gliksbot.axonhome.core.sim.SIM_EPOCH_SECONDS + 86_400L
    private fun now(): Instant = Instant.fromEpochSeconds(clockSeconds++)

    /** Retained field snapshots so snapshots stay addressable by field_id. */
    private val snapshotHistory = LinkedHashMap<String, SharedFieldSnapshot>()

    /** Exported capsules kept for later verification. */
    private val capsulesById = LinkedHashMap<String, RecoveryCapsule>()
    private val capsuleVerifiedAt = mutableMapOf<String, Instant>()

    /** Pending two-step mask confirmation (single outstanding proposal). */
    private var pendingMaskToken: String? = null
    private var pendingMaskPolicies: Map<LogicalRegion, MaskPolicy>? = null

    /** Roundtable messages posted via the API (plus sim ambient chatter). */
    private val agentMessages = mutableListOf<AgentMessage>()

    /** Capture the current head snapshot if it is new (called before reads). */
    @Synchronized
    fun syncHead() {
        val head = engine.field
        snapshotHistory.putIfAbsent(head.fieldId, head)
    }

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------

    private fun coreRegionOf(region: LogicalRegion): CoreRegion = CoreRegion.valueOf(region.name)
    private fun cpRegionOf(region: CoreRegion): LogicalRegion = LogicalRegion.valueOf(region.name)

    private fun heartSummary(): HeartHealthSummary {
        val h = engine.health()
        return HeartHealthSummary(
            heartEpochId = h.epochId,
            heartbeatSequence = h.heartbeatSequence,
            tickSequence = h.tickSequence,
            headFieldId = h.headFieldId,
            tickInFlight = h.tickInFlight,
            lastBeatOk = true,
            lastFailure = h.lastFailure,
            maskRevision = h.maskRevision,
            lastViewId = h.lastViewId,
            dormantIndexId = "sim-dormant-index",
        )
    }

    private fun maskPolicyToCp(policy: RegionMaskPolicy): MaskPolicy = when (policy.kind) {
        RegionMaskPolicy.KIND_ALL -> MaskPolicy(MaskKind.ALL)
        RegionMaskPolicy.KIND_NONE -> MaskPolicy(MaskKind.NONE)
        RegionMaskPolicy.KIND_LAST_N_SPANS -> MaskPolicy(MaskKind.LAST_N_SPANS, n = policy.limit)
        RegionMaskPolicy.KIND_TAIL_PERCENT -> MaskPolicy(MaskKind.TAIL_PERCENT, tailPercent = policy.limit)
        else -> MaskPolicy(MaskKind.NONE)
    }

    private fun maskPolicyToCore(policy: MaskPolicy): RegionMaskPolicy = when (policy.kind) {
        MaskKind.ALL -> RegionMaskPolicy.all()
        MaskKind.NONE -> RegionMaskPolicy.none()
        MaskKind.LAST_N_SPANS -> RegionMaskPolicy.lastNSpans(policy.n ?: 0)
        MaskKind.TAIL_PERCENT -> RegionMaskPolicy.tailPercent(policy.tailPercent ?: 0)
    }

    private fun maskStateToCp(state: com.gliksbot.axonhome.core.domain.HeartRegionMaskState): MaskState =
        MaskState(
            freshness = Freshness.LIVE,
            observedAt = now(),
            revision = state.revision,
            stateId = state.stateId,
            viewId = engine.health().lastViewId,
            policies = com.gliksbot.axonhome.core.domain.CANONICAL_REGION_ORDER.associate { region ->
                cpRegionOf(region) to maskPolicyToCp(state.policyFor(region))
            },
        )

    private fun coreStatusToCp(status: com.gliksbot.axonhome.core.domain.CoreStatus): CoreStatus =
        CoreStatus.valueOf(status.name)

    // ------------------------------------------------------------------
    // health / field
    // ------------------------------------------------------------------

    override suspend fun health(): HealthResponse {
        syncHead()
        return HealthResponse(
            freshness = Freshness.LIVE,
            observedAt = now(),
            controlPlane = ControlPlaneStatus(
                status = if (engine.organismPaused) "degraded" else "ok",
                version = "sim-0.1.0",
                apiVersions = listOf("v1"),
                uptimeSeconds = clockSeconds - com.gliksbot.axonhome.core.sim.SIM_EPOCH_SECONDS,
            ),
            heart = heartSummary(),
        )
    }

    override suspend fun fieldHead(): FieldHead {
        syncHead()
        val f = engine.field
        return FieldHead(
            freshness = Freshness.LIVE,
            observedAt = now(),
            head = BranchHead(
                generation = f.tickId.toLong(),
                fieldId = f.fieldId,
                tickId = f.tickId.toLong(),
                parentFieldId = f.parentFieldId,
            ),
            health = heartSummary(),
        )
    }

    override suspend fun fieldSnapshot(fieldId: String): FieldSnapshotView {
        syncHead()
        val snapshot = snapshotHistory[fieldId]
            ?: throw ApiException(
                ApiError(
                    code = ErrorCode.NOT_FOUND,
                    message = "snapshot $fieldId not retained by the simulator",
                    retryable = false,
                )
            )
        return FieldSnapshotView(
            freshness = Freshness.LIVE,
            observedAt = now(),
            fieldId = snapshot.fieldId,
            tickId = snapshot.tickId.toLong(),
            parentFieldId = snapshot.parentFieldId,
            regions = snapshot.regions.map { region ->
                RegionSummary(
                    region = cpRegionOf(region.name),
                    spanCount = region.spans.size.toLong(),
                    charCount = region.text.length.toLong(),
                    visibility = RegionVisibility.ATTENDED,
                    writePolicy = if (region.name in CORE_WRITABLE_REGIONS) {
                        RegionWritePolicy.CORE_WRITABLE
                    } else {
                        RegionWritePolicy.SEALED
                    },
                    regionSha256 = region.canonicalHash,
                )
            },
            canonicalSha256 = snapshot.canonicalHash,
        )
    }

    override suspend fun fieldTicks(count: Int): TickList {
        syncHead()
        // The branch journal only ever records commits; Heart-rejected ticks
        // never enter it (visible instead as alerts in the event stream).
        val ticks = engine.tickHistory.filter { !it.rejected }.takeLast(count).map { record ->
            TickRecord(
                event = BranchEventKind.COMMIT,
                generation = record.tickNumber.toLong(),
                tickId = record.tickNumber.toLong(),
                baseFieldId = record.baseFieldId,
                successorFieldId = record.resultFieldId ?: record.baseFieldId,
                deltaIds = record.deltaIds,
                heartCommitId = record.deltaIds.lastOrNull(),
                consolidatorCoreId = record.consolidatorCoreId,
                timestamp = now(),
            )
        }
        return TickList(
            freshness = Freshness.LIVE,
            observedAt = now(),
            ticks = ticks,
        )
    }

    // ------------------------------------------------------------------
    // masks (two-step confirm)
    // ------------------------------------------------------------------

    override suspend fun masks(): MaskState = maskStateToCp(engine.maskState)

    override suspend fun putMasks(request: PutMasksRequest, idempotencyKey: String): PutMasksResponse {
        if (!request.confirmed) {
            // First touch: build a PROPOSED state; nothing is mutated.
            var proposed = engine.maskState
            for ((region, policy) in request.policies) {
                proposed = proposed.withPolicy(coreRegionOf(region), maskPolicyToCore(policy))
            }
            val token = "sim-confirm-" + canonicalSha256("${proposed.stateId}|$idempotencyKey").take(16)
            pendingMaskToken = token
            pendingMaskPolicies = request.policies
            engine.auditLog.record(
                who = "operator", device = "android-app", command = "mask_proposal",
                target = "region_masks", timestamp = now().toString(),
                previousState = engine.maskState.stateId, requestedState = proposed.stateId,
                result = "proposed",
            )
            return PutMasksResponse(
                freshness = Freshness.LIVE,
                observedAt = now(),
                applied = false,
                confirmToken = token,
                proposed = maskStateToCp(proposed),
            )
        }
        if (request.confirmToken == null || request.confirmToken != pendingMaskToken) {
            throw ApiException(
                ApiError(
                    code = ErrorCode.MASK_CONFIRMATION_REQUIRED,
                    message = "missing or stale confirm token; request a fresh proposal first",
                    retryable = false,
                )
            )
        }
        val policies = pendingMaskPolicies.orEmpty()
        for ((region, policy) in policies) {
            engine.setRegionMaskPolicy(coreRegionOf(region), maskPolicyToCore(policy))
        }
        pendingMaskToken = null
        pendingMaskPolicies = null
        return PutMasksResponse(
            freshness = Freshness.LIVE,
            observedAt = now(),
            applied = true,
            confirmToken = null,
            proposed = maskStateToCp(engine.maskState),
        )
    }

    // ------------------------------------------------------------------
    // cores / souls
    // ------------------------------------------------------------------

    override suspend fun cores(): CoreList = CoreList(
        freshness = Freshness.LIVE,
        observedAt = now(),
        cores = engine.cores.map { simCore ->
            val soul = engine.soulOf(simCore.descriptor.coreId)
            CoreDescriptor(
                coreId = simCore.descriptor.coreId,
                status = coreStatusToCp(simCore.descriptor.status),
                architectureId = simCore.descriptor.architectureId,
                parameterGeneration = simCore.descriptor.parameterGeneration,
                writableRegions = simCore.descriptor.writableRegions.map { cpRegionOf(it) },
                soulId = soul.soulId,
                soulGeneration = soul.generation.toLong(),
                mode = if (simCore.consolidatorEligible) "participant+consolidator" else "participant",
            )
        },
    )

    override suspend fun core(coreId: String): CoreDetail {
        val simCore = engine.cores.firstOrNull { it.descriptor.coreId == coreId }
            ?: throw ApiException(ApiError(ErrorCode.NOT_FOUND, "core '$coreId' not found", retryable = false))
        val soul = engine.soulOf(coreId)
        return CoreDetail(
            freshness = Freshness.LIVE,
            observedAt = now(),
            core = CoreDescriptor(
                coreId = simCore.descriptor.coreId,
                status = coreStatusToCp(simCore.descriptor.status),
                architectureId = simCore.descriptor.architectureId,
                parameterGeneration = simCore.descriptor.parameterGeneration,
                writableRegions = simCore.descriptor.writableRegions.map { cpRegionOf(it) },
                soulId = soul.soulId,
                soulGeneration = soul.generation.toLong(),
                mode = if (simCore.consolidatorEligible) "participant+consolidator" else "participant",
            ),
            anatomy = CoreAnatomy(
                dModel = simCore.anatomy.dModel,
                nHeads = simCore.anatomy.nHeads,
                nLayers = simCore.anatomy.nLayers,
                ffnDim = simCore.anatomy.ffnDim,
                stateTokens = simCore.anatomy.stateTokens,
                pageSize = simCore.anatomy.pageSize,
            ),
        )
    }

    override suspend fun soul(coreId: String): SoulSummary {
        val soul = engine.soulOf(coreId)
        val receipts = engine.soulReceiptsOf(coreId).takeLast(8).map { receipt ->
            SoulCommitReceiptMeta(
                receiptId = receipt.receiptId,
                transitionId = receipt.transitionId,
                beforeSoulId = receipt.beforeSoulId,
                afterSoulId = receipt.afterSoulId,
                generation = receipt.generation.toLong(),
                phase = when (receipt.phase) {
                    "first" -> SoulPhase.FIRST
                    "refined" -> SoulPhase.REFINED
                    "consolidated" -> SoulPhase.CONSOLIDATED
                    "outcome" -> SoulPhase.OUTCOME
                    "training" -> SoulPhase.TRAINING
                    else -> SoulPhase.MERGE
                },
                tickUid = receipt.tickUid,
                commitBinding = receipt.commitBinding,
            )
        }
        return SoulSummary(
            freshness = Freshness.LIVE,
            observedAt = now(),
            coreId = coreId,
            branchId = "live",
            head = SoulHead(
                soulId = soul.soulId,
                generation = soul.generation.toLong(),
                parentSoulId = soul.parentSoulId,
                parameterGeneration = soul.parameterGeneration,
                architectureId = soul.architectureId,
            ),
            layers = soul.layers.map { layer ->
                SoulLayerMeta(
                    temperature = com.gliksbot.axonhome.controlplane.SoulTemperature.valueOf(layer.temperature.name),
                    mediaType = layer.mediaType,
                    payloadBytes = layer.payloadBytes,
                    payloadSha256 = layer.payloadSha256,
                )
            },
            receiptsTail = receipts,
        )
    }

    // ------------------------------------------------------------------
    // trainer
    // ------------------------------------------------------------------

    override suspend fun trainerStatus(): TrainerStatus {
        val trainer = engine.trainer
        val generations = if (trainer == null) emptyList() else listOf(
            ActiveGeneration(
                moduleId = trainer.config.moduleId,
                candidateGenerationId = trainer.candidateGenerationId,
                baseGenerationId = trainer.baseGenerationId,
                lifecycleStatus = when (trainer.lifecycle) {
                    com.gliksbot.axonhome.core.domain.CandidateLifecycleStatus.RUNNING -> TrainerLifecycleStatus.RUNNING
                    com.gliksbot.axonhome.core.domain.CandidateLifecycleStatus.PAUSED -> TrainerLifecycleStatus.PAUSED
                    com.gliksbot.axonhome.core.domain.CandidateLifecycleStatus.COMPLETED -> TrainerLifecycleStatus.COMPLETED
                    else -> TrainerLifecycleStatus.PREPARED
                },
                currentStep = trainer.globalStep.toLong(),
                latestLoss = if (trainer.lastLoss.isNaN()) null else trainer.lastLoss,
                learningRate = trainer.policy.learningRate,
            )
        )
        val checkpoints = trainer?.retainedCheckpoints.orEmpty().map { checkpoint ->
            CheckpointRecordMeta(
                checkpointId = checkpoint.checkpointId,
                moduleId = checkpoint.moduleId,
                candidateGenerationId = checkpoint.candidateGenerationId,
                step = checkpoint.step.toLong(),
                artifactRelpath = checkpoint.artifactRelpath,
                artifactSha256 = checkpoint.artifactSha256,
                artifactBytes = checkpoint.artifactBytes,
                optimizerIncluded = checkpoint.optimizerIncluded,
            )
        }
        return TrainerStatus(
            freshness = Freshness.LIVE,
            observedAt = now(),
            inspection = TrainerInspection(
                writerLeasePresent = trainer != null,
                leaseOwner = if (trainer != null) "sim-trainer" else null,
                activeGenerations = generations,
                latestCheckpoints = checkpoints,
                availableCommands = mapOf(
                    TrainerCommandKind.STATUS to true,
                    TrainerCommandKind.PREFLIGHT to true,
                    TrainerCommandKind.START to true,
                    TrainerCommandKind.PAUSE to true,
                    TrainerCommandKind.RESUME to true,
                    TrainerCommandKind.EVALUATE to false,
                    TrainerCommandKind.INVENTORY to false,
                    TrainerCommandKind.CONFIGURE to false,
                    TrainerCommandKind.EXPORT_CLOUD_PACKET to false,
                    TrainerCommandKind.IMPORT_CLOUD_RESULT to false,
                    TrainerCommandKind.COMPARE to false,
                    TrainerCommandKind.REQUEST_PROMOTION to false,
                    TrainerCommandKind.ROLLBACK to false,
                ),
            ),
        )
    }

    override suspend fun trainerCommand(request: TrainerCommandRequest): TrainerCommandResponse {
        val commandId = "sim-cmd-" + canonicalSha256("${request.kind}|${request.idempotencyKey}").take(16)
        fun respond(status: TrainerCommandStatus, message: String): TrainerCommandResponse {
            engine.auditLog.record(
                who = request.requestedBy, device = "android-app",
                command = "trainer_command:${request.kind.name.lowercase()}",
                target = engine.trainer?.candidateGenerationId ?: "no-candidate",
                timestamp = now().toString(),
                previousState = engine.trainer?.lifecycle?.name ?: "none",
                requestedState = request.kind.name,
                result = status.name,
            )
            return TrainerCommandResponse(
                freshness = Freshness.LIVE,
                observedAt = now(),
                commandId = commandId,
                kind = request.kind,
                status = status,
                message = message,
                auditId = commandId,
            )
        }
        return when (request.kind) {
            TrainerCommandKind.STATUS -> respond(TrainerCommandStatus.OK, "status snapshot current")
            TrainerCommandKind.PREFLIGHT -> respond(
                TrainerCommandStatus.OK,
                "preflight ok: anatomy locked to d64, optimizer adamw, tranche ledger ready [SIMULATION]",
            )
            TrainerCommandKind.START -> {
                if (engine.trainer != null &&
                    engine.trainer!!.lifecycle == com.gliksbot.axonhome.core.domain.CandidateLifecycleStatus.RUNNING
                ) {
                    respond(TrainerCommandStatus.REJECTED, "a candidate is already running")
                } else if (engine.trainer == null) {
                    engine.startTraining()
                    respond(TrainerCommandStatus.OK, "candidate started")
                } else {
                    engine.requestNextTranche()
                    respond(TrainerCommandStatus.OK, "training resumed on a fresh tranche")
                }
            }
            TrainerCommandKind.PAUSE -> {
                val result = engine.stopTraining(StopKind.SAFE)
                when (result) {
                    is StopResult.Safe -> respond(
                        TrainerCommandStatus.OK,
                        "paused at boundary ${result.result.boundary}; preserved ${result.result.preserved.size} records",
                    )
                    is StopResult.Emergency -> respond(TrainerCommandStatus.FAILED, "unexpected emergency result")
                }
            }
            TrainerCommandKind.RESUME -> {
                val trainer = engine.trainer
                if (trainer == null) {
                    respond(TrainerCommandStatus.REJECTED, "no candidate to resume; start first")
                } else {
                    try {
                        engine.requestNextTranche()
                        respond(TrainerCommandStatus.OK, "resumed on next tranche")
                    } catch (e: IllegalArgumentException) {
                        respond(TrainerCommandStatus.REJECTED, e.message ?: "resume rejected")
                    }
                }
            }
            else -> respond(
                TrainerCommandStatus.UNAVAILABLE,
                "no governed handler for ${request.kind.name.lowercase()}; no action was taken",
            )
        }
    }

    // ------------------------------------------------------------------
    // compute
    // ------------------------------------------------------------------

    private fun workerToCp(worker: com.gliksbot.axonhome.core.sim.SimWorker): Worker = Worker(
        workerId = worker.workerId,
        provider = worker.provider,
        endpoint = "sim://workers/${worker.workerId}",
        label = worker.workerId,
        status = if (worker.connected) WorkerStatus.CONNECTED else WorkerStatus.LOST,
        capabilities = CapabilitySet(
            supportsStatus = true,
            supportsStop = worker.connected,
            supportsGpuMetrics = true,
        ),
        accelerator = worker.accelerator,
        lastHeartbeatAt = if (worker.connected) now() else null,
        activeJobId = if (engine.trainer != null && worker.connected) {
            engine.trainer!!.candidateGenerationId
        } else {
            null
        },
        workerVersion = "sim-worker-1.0",
        gitCommit = "sim",
        registeredAt = Instant.fromEpochSeconds(com.gliksbot.axonhome.core.sim.SIM_EPOCH_SECONDS),
    )

    override suspend fun workers(): WorkerList = WorkerList(
        freshness = Freshness.LIVE,
        observedAt = now(),
        workers = engine.workers.map { workerToCp(it) },
    )

    override suspend fun registerWorker(request: RegisterWorkerRequest, idempotencyKey: String): Worker {
        val workerId = "sim-worker-" + canonicalSha256("${request.endpoint}|$idempotencyKey").take(8)
        engine.connectWorker(workerId, request.provider, request.accelerator ?: "unknown")
        engine.auditLog.record(
            who = "operator", device = "android-app", command = "register_worker",
            target = workerId, timestamp = now().toString(),
            previousState = "absent", requestedState = "connected", result = "connected",
        )
        return workerToCp(engine.workers.first { it.workerId == workerId })
    }

    override suspend fun stopWorker(workerId: String, idempotencyKey: String): WorkerStopResponse {
        val worker = engine.workers.firstOrNull { it.workerId == workerId }
            ?: throw ApiException(ApiError(ErrorCode.NOT_FOUND, "worker '$workerId' not found", retryable = false))
        val ack = if (worker.connected) {
            engine.loseWorker(workerId, "operator stop via Axon Home")
            ComponentAck(
                componentId = workerId,
                componentKind = ComponentKind.COMPUTE_WORKER,
                status = ComponentAckStatus.CONFIRMED_STOPPED,
                detail = "worker acknowledged stop",
            )
        } else {
            ComponentAck(
                componentId = workerId,
                componentKind = ComponentKind.COMPUTE_WORKER,
                status = ComponentAckStatus.UNREACHABLE,
                detail = "worker already unreachable; stop not confirmed",
            )
        }
        return WorkerStopResponse(
            freshness = Freshness.LIVE,
            observedAt = now(),
            workerId = workerId,
            ack = ack,
        )
    }

    // ------------------------------------------------------------------
    // storage
    // ------------------------------------------------------------------

    override suspend fun artifacts(): ArtifactList {
        val checkpoints = engine.trainer?.retainedCheckpoints.orEmpty().map { checkpoint ->
            Artifact(
                artifactId = checkpoint.checkpointId,
                kind = ArtifactKind.CHECKPOINT,
                provider = "local",
                path = checkpoint.artifactRelpath,
                sha256 = checkpoint.artifactSha256,
                bytes = checkpoint.artifactBytes,
                createdAt = now(),
                contentAddressed = true,
                verifiedAt = null,
            )
        }
        val capsuleArtifacts = capsulesById.values.map { capsule ->
            Artifact(
                artifactId = capsule.manifest.capsuleId,
                kind = ArtifactKind.CAPSULE,
                provider = "local",
                path = "capsules/${capsule.manifest.capsuleId}.json",
                sha256 = capsule.manifest.manifestSha256,
                bytes = capsule.files.values.sumOf { it.size.toLong() },
                createdAt = now(),
                contentAddressed = true,
                verifiedAt = capsuleVerifiedAt[capsule.manifest.capsuleId],
            )
        }
        return ArtifactList(
            freshness = Freshness.LIVE,
            observedAt = now(),
            artifacts = checkpoints + capsuleArtifacts,
        )
    }

    override suspend fun registerArtifact(request: RegisterArtifactRequest, idempotencyKey: String): Artifact =
        Artifact(
            artifactId = "sim-artifact-" + canonicalSha256("${request.path}|$idempotencyKey").take(8),
            kind = request.kind,
            provider = request.provider,
            path = request.path,
            sha256 = request.sha256,
            bytes = request.bytes,
            createdAt = now(),
            contentAddressed = request.contentAddressed,
        )

    // ------------------------------------------------------------------
    // capsules
    // ------------------------------------------------------------------

    override suspend fun capsules(): CapsuleList = CapsuleList(
        freshness = Freshness.LIVE,
        observedAt = now(),
        capsules = capsulesById.values.map { capsule ->
            CapsuleSummary(
                capsuleId = capsule.manifest.capsuleId,
                kind = CapsuleKind.FULL_ORGANISM,
                archiveSha256 = capsule.manifest.manifestSha256,
                archiveBytes = capsule.files.values.sumOf { it.size.toLong() },
                createdAt = now(),
                verificationStatus = if (capsuleVerifiedAt.containsKey(capsule.manifest.capsuleId)) {
                    CapsuleVerificationStatus.VERIFIED
                } else {
                    CapsuleVerificationStatus.UNVERIFIED
                },
            )
        },
    )

    override suspend fun exportCapsule(request: CapsuleExportRequest, idempotencyKey: String): CapsuleManifest {
        val capsule = try {
            engine.exportRecoveryCapsule()
        } catch (e: IllegalStateException) {
            throw ApiException(
                ApiError(ErrorCode.CONFLICT, e.message ?: "capsule export unavailable", retryable = false)
            )
        }
        capsulesById[capsule.manifest.capsuleId] = capsule
        val trainer = engine.trainer
        return CapsuleManifest(
            capsuleId = capsule.manifest.capsuleId,
            kind = CapsuleKind.FULL_ORGANISM,
            archiveName = "${capsule.manifest.capsuleId}.tar",
            archiveSha256 = capsule.manifest.manifestSha256,
            archiveBytes = capsule.files.values.sumOf { it.size.toLong() },
            members = capsule.manifest.members.mapValues { (_, member) ->
                ManifestMember(sha256 = member.sha256, bytes = member.bytes)
            },
            architectureId = trainer?.anatomy?.architectureId,
            baseGenerationId = trainer?.baseGenerationId,
            candidateGenerationId = trainer?.candidateGenerationId,
            checkpointId = trainer?.latestCheckpoint?.checkpointId,
            soulIds = trainer?.let { listOf(it.candidateSoul.soulId) }.orEmpty(),
            gitCommit = "sim-local",
            createdAt = now(),
            storage = CapsuleStorage(provider = "local", path = "capsules/${capsule.manifest.capsuleId}"),
        )
    }

    override suspend fun verifyCapsule(capsuleId: String, idempotencyKey: String): CapsuleVerificationReport {
        val capsule = capsulesById[capsuleId]
            ?: throw ApiException(ApiError(ErrorCode.NOT_FOUND, "capsule '$capsuleId' not found", retryable = false))
        val verification = RecoveryCapsuleVerifier.verify(
            files = capsule.files,
            manifest = capsule.manifest,
            expectedArchitectureId = engine.trainer?.anatomy?.architectureId,
            expectedCandidateGenerationId = engine.trainer?.candidateGenerationId,
        )
        val allChecks = CapsuleCheckName.entries.toList()
        val passedNames = verification.checksPassed.toSet()
        val nameMap = mapOf(
            "manifest-schema" to CapsuleCheckName.MEMBER_SET, // schema check folded into member set view
            "member-set" to CapsuleCheckName.MEMBER_SET,
            "member-hashes" to CapsuleCheckName.MEMBER_SHA256,
            "member-decode" to CapsuleCheckName.MEMBER_SHA256,
            "architecture-identity" to CapsuleCheckName.ARCHITECTURE_IDENTITY,
            "candidate-generation" to CapsuleCheckName.CANDIDATE_GENERATION,
            "optimizer-policy" to CapsuleCheckName.OPTIMIZER_POLICY,
            "soul-checkpoint-pairing" to CapsuleCheckName.SOUL_CHECKPOINT_PAIRING,
            "lineage" to CapsuleCheckName.LINEAGE,
        )
        val mappedPassed = verification.checksPassed.mapNotNull { nameMap[it] }.toSet()
        val checks = allChecks.map { name ->
            CapsuleCheck(
                name = name,
                ok = verification.ok && (name in mappedPassed || name == CapsuleCheckName.ARCHIVE_SHA256),
                detail = when (name) {
                    CapsuleCheckName.ARCHIVE_SHA256 ->
                        "manifest sha256 ${capsule.manifest.manifestSha256.take(16)}… (sim capsule, no separate archive)"
                    else -> null
                },
            )
        }
        if (verification.ok) {
            capsuleVerifiedAt[capsuleId] = now()
        }
        engine.auditLog.record(
            who = "operator", device = "android-app", command = "verify_capsule",
            target = capsuleId, timestamp = now().toString(),
            previousState = "unverified",
            requestedState = "verified",
            result = if (verification.ok) "verified" else "quarantined",
            error = verification.mismatches.firstOrNull(),
        )
        return CapsuleVerificationReport(
            capsuleId = capsuleId,
            ok = verification.ok,
            checks = checks,
            mismatches = verification.mismatches,
            quarantineDir = if (verification.ok) null else "sim://quarantine/$capsuleId",
            verifiedAt = now(),
        )
    }

    // ------------------------------------------------------------------
    // audit
    // ------------------------------------------------------------------

    override suspend fun audit(limit: Int, before: Instant?): AuditList {
        val entries = engine.auditLog.entries
            .filter { entry -> before == null || runCatching { Instant.parse(entry.timestamp) < before }.getOrDefault(true) }
            .takeLast(limit)
            .reversed()
            .mapIndexed { index, entry ->
                AuditEntry(
                    auditId = "sim-audit-" + canonicalSha256("${entry.timestamp}|${entry.command}|$index").take(12),
                    who = entry.who,
                    device = entry.device,
                    command = entry.command,
                    target = entry.target,
                    timestamp = runCatching { Instant.parse(entry.timestamp) }
                        .getOrElse { Instant.fromEpochSeconds(com.gliksbot.axonhome.core.sim.SIM_EPOCH_SECONDS) },
                    previousState = entry.previousState,
                    requestedState = entry.requestedState,
                    result = when (entry.result.lowercase()) {
                        "committed", "connected", "verified", "ok", "ticking", "paused",
                        "boundary-preserved", "all-stopped", "published", "proposed" -> AuditResult.SUCCESS
                        "partial" -> AuditResult.PARTIAL
                        else -> AuditResult.SUCCESS
                    },
                    error = entry.error,
                )
            }
        return AuditList(freshness = Freshness.LIVE, observedAt = now(), entries = entries)
    }

    // ------------------------------------------------------------------
    // agents
    // ------------------------------------------------------------------

    override suspend fun agents(): AgentList = AgentList(
        freshness = Freshness.LIVE,
        observedAt = now(),
        agents = engine.agents.map { agentId ->
            AgentRecord(
                agentId = agentId,
                name = agentId,
                provider = "sim-roundtable",
                model = "sim-model-$agentId",
                online = true,
                role = "roundtable participant",
                activeTask = null,
                capabilities = AgentCapabilities(
                    supportsStreaming = false,
                    supportsTools = false,
                    supportsSystemPrompt = true,
                    supportsModelList = false,
                ),
                vaultKeyRef = null,
            )
        },
    )

    override suspend fun registerAgent(request: RegisterAgentRequest, idempotencyKey: String): AgentRecord {
        engine.auditLog.record(
            who = "operator", device = "android-app", command = "register_agent",
            target = request.name, timestamp = now().toString(),
            previousState = "local", requestedState = "registered", result = "recorded-locally",
        )
        return AgentRecord(
            agentId = "local-" + canonicalSha256("${request.name}|$idempotencyKey").take(8),
            name = request.name,
            provider = request.provider,
            model = request.model,
            online = false,
            role = request.role,
            vaultKeyRef = request.vaultKeyRef,
        )
    }

    override suspend fun postAgentMessage(
        agentId: String,
        request: AgentMessageRequest,
        idempotencyKey: String,
    ): AgentMessage {
        engine.receiveAgentMessage(agentId, request.body)
        val message = AgentMessage(
            messageId = "sim-msg-" + canonicalSha256("$agentId|${request.body}|$idempotencyKey").take(12),
            agentId = agentId,
            from = "operator",
            body = request.body,
            sentAt = now(),
            inReplyTo = request.inReplyTo,
        )
        agentMessages += message
        return message
    }

    /** Recent Roundtable chatter (ambient sim messages + operator posts). */
    fun recentAgentMessages(): List<AgentMessage> = agentMessages.toList()

    // ------------------------------------------------------------------
    // stop
    // ------------------------------------------------------------------

    override suspend fun stop(request: StopRequest, idempotencyKey: String): StopResponse {
        return when (request.scope) {
            StopScope.PAUSE_AXON -> {
                val result = engine.pauseOrganism()
                StopResponse(
                    freshness = Freshness.LIVE,
                    observedAt = now(),
                    scope = request.scope,
                    allStopped = true,
                    components = listOf(
                        ComponentAck(
                            componentId = "heart",
                            componentKind = ComponentKind.HEART_HOST,
                            status = ComponentAckStatus.CONFIRMED_STOPPED,
                            detail = "paused at boundary ${result.boundary}; preserved ${result.preserved.size} records",
                        )
                    ),
                )
            }
            StopScope.STOP_TRAINING -> {
                val result = engine.stopTraining(StopKind.SAFE)
                val detail = when (result) {
                    is StopResult.Safe ->
                        "stopped at boundary ${result.result.boundary}; preserved: ${result.result.preserved.joinToString()}"
                    is StopResult.Emergency -> "unexpected emergency result"
                }
                StopResponse(
                    freshness = Freshness.LIVE,
                    observedAt = now(),
                    scope = request.scope,
                    allStopped = true,
                    components = listOf(
                        ComponentAck(
                            componentId = engine.trainer?.candidateGenerationId ?: "no-candidate",
                            componentKind = ComponentKind.TRAINER,
                            status = ComponentAckStatus.CONFIRMED_STOPPED,
                            detail = detail,
                        )
                    ),
                )
            }
            StopScope.STOP_COMPUTE -> {
                val workerId = request.target
                    ?: throw ApiException(ApiError(ErrorCode.VALIDATION_FAILED, "STOP_COMPUTE requires a worker target", retryable = false))
                val response = stopWorker(workerId, idempotencyKey)
                StopResponse(
                    freshness = Freshness.LIVE,
                    observedAt = now(),
                    scope = request.scope,
                    allStopped = response.ack.status == ComponentAckStatus.CONFIRMED_STOPPED,
                    components = listOf(response.ack),
                )
            }
            StopScope.EMERGENCY -> {
                val result = engine.emergencyStopAll()
                val components = result.acknowledgements.map { (componentId, ack) ->
                    ComponentAck(
                        componentId = componentId,
                        componentKind = when {
                            componentId == "heart" -> ComponentKind.HEART_HOST
                            componentId.startsWith("trainer:") -> ComponentKind.TRAINER
                            componentId.startsWith("worker:") -> ComponentKind.COMPUTE_WORKER
                            else -> ComponentKind.COMPUTE_WORKER
                        },
                        status = when (ack) {
                            com.gliksbot.axonhome.core.stop.ComponentAck.STOPPED -> ComponentAckStatus.CONFIRMED_STOPPED
                            com.gliksbot.axonhome.core.stop.ComponentAck.UNREACHABLE -> ComponentAckStatus.UNREACHABLE
                            com.gliksbot.axonhome.core.stop.ComponentAck.FAILED -> ComponentAckStatus.REFUSED
                        },
                        detail = if (ack == com.gliksbot.axonhome.core.stop.ComponentAck.UNREACHABLE) {
                            "component unreachable; stop NOT confirmed"
                        } else {
                            null
                        },
                    )
                }
                // Never trust a bare claim: derive allStopped from the ack map.
                StopResponse(
                    freshness = Freshness.LIVE,
                    observedAt = now(),
                    scope = request.scope,
                    allStopped = StopAccounting.computeAllStopped(components),
                    components = components,
                )
            }
        }
    }

    // ------------------------------------------------------------------
    // events
    // ------------------------------------------------------------------

    /**
     * Cold stream of the typed event envelope. The simulator is driven by
     * [SimulationDriver]; this flow replays events appended after [lastSeq]
     * and then tails the log. Emitted events are real sim events; there are
     * no synthetic heartbeats.
     */
    override fun events(lastSeq: Long?): Flow<AxonEvent> = flow {
        var cursor = (lastSeq ?: -1L) + 1
        while (true) {
            val batch = engine.eventLog.replayFrom(cursor)
            for (event in batch) {
                toControlplaneEvent(event)?.let { emit(it) }
                cursor = event.seq + 1
            }
            kotlinx.coroutines.delay(250)
        }
    }

    /** Map one core event to the typed control-plane envelope. */
    private fun toControlplaneEvent(event: com.gliksbot.axonhome.core.events.AxonEvent): AxonEvent? {
        val timestamp = runCatching { Instant.parse(event.timestamp) }
            .getOrElse { Instant.fromEpochSeconds(com.gliksbot.axonhome.core.sim.SIM_EPOCH_SECONDS) }
        return when (val payload = event.payload) {
            is AxonEventPayload.TickStarted -> AxonEvent.TickStarted(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = TickStartedPayload(
                    tickSequence = payload.tickId.toLong(),
                    heartbeatId = "sim-hb-${event.seq}",
                    baseFieldId = payload.baseFieldId,
                    baseTickId = (payload.tickId - 1).toLong(),
                    viewId = engine.health().lastViewId,
                    participants = engine.activeCores.map { it.descriptor.coreId },
                ),
            )
            is AxonEventPayload.CoreFirstReturned -> AxonEvent.CoreFirstReturned(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = ProposalReturnedPayload(
                    tickSequence = payload.tickId.toLong(),
                    coreId = payload.coreId,
                    proposalId = payload.proposalId,
                    participantState = ParticipantState.RETURNED,
                ),
            )
            is AxonEventPayload.FirstBarrierComplete -> AxonEvent.FirstBarrierComplete(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = BarrierCompletePayload(
                    tickSequence = payload.tickId.toLong(),
                    workspaceId = "sim-workspace-${payload.tickId}-first",
                    pass = "first",
                    returnedCoreIds = payload.participants,
                ),
            )
            is AxonEventPayload.CoreRefinedReturned -> AxonEvent.CoreRefinedReturned(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = ProposalReturnedPayload(
                    tickSequence = payload.tickId.toLong(),
                    coreId = payload.coreId,
                    proposalId = payload.proposalId,
                    participantState = ParticipantState.RETURNED,
                ),
            )
            is AxonEventPayload.RefinedBarrierComplete -> AxonEvent.RefinedBarrierComplete(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = BarrierCompletePayload(
                    tickSequence = payload.tickId.toLong(),
                    workspaceId = "sim-workspace-${payload.tickId}-refined",
                    pass = "refined",
                    returnedCoreIds = payload.participants,
                ),
            )
            is AxonEventPayload.ConsolidatorSelected -> AxonEvent.ConsolidatorSelected(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = ConsolidatorSelectedPayload(
                    tickSequence = payload.tickId.toLong(),
                    coreId = payload.coreId,
                    participantCount = engine.activeCores.size,
                ),
            )
            is AxonEventPayload.FinalReturned -> AxonEvent.FinalReturned(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = FinalReturnedPayload(
                    tickSequence = payload.tickId.toLong(),
                    consolidatorCoreId = payload.coreId,
                    verdictId = payload.verdictId,
                    regionTags = listOf("response_draft", "scratch"),
                ),
            )
            is AxonEventPayload.HeartValidationStarted -> AxonEvent.HeartValidationStarted(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = HeartValidationStartedPayload(
                    tickSequence = payload.tickId.toLong(),
                    frozenImageId = "sim-frozen-${payload.tickId}",
                    viewId = engine.health().lastViewId,
                ),
            )
            is AxonEventPayload.FieldCommitted -> AxonEvent.FieldCommitted(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = FieldCommittedPayload(
                    tickSequence = payload.tickId.toLong(),
                    baseFieldId = payload.parentFieldId ?: payload.fieldId,
                    successorFieldId = payload.fieldId,
                    deltaIds = payload.deltaIds,
                    commitId = "sim-commit-${event.seq}",
                    journalGeneration = payload.tickId.toLong(),
                ),
            )
            is AxonEventPayload.SoulTransitionAccepted -> AxonEvent.SoulTransitionAccepted(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = SoulTransitionAcceptedPayload(
                    coreId = payload.coreId,
                    receiptId = payload.receiptId,
                    beforeSoulId = "",
                    afterSoulId = payload.soulId,
                    generation = payload.generation.toLong(),
                    phase = when (payload.phase) {
                        "first" -> SoulPhase.FIRST
                        "refined" -> SoulPhase.REFINED
                        "consolidated" -> SoulPhase.CONSOLIDATED
                        "outcome" -> SoulPhase.OUTCOME
                        "training" -> SoulPhase.TRAINING
                        else -> SoulPhase.MERGE
                    },
                ),
            )
            is AxonEventPayload.TrainerStepAccepted -> AxonEvent.TrainerStepAccepted(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = TrainerStepAcceptedPayload(
                    bundleId = payload.bundleId,
                    pointerId = engine.trainer?.pointer?.pointerId ?: "",
                    moduleId = payload.moduleId,
                    candidateGenerationId = payload.candidateGenerationId,
                    coreId = engine.trainer?.config?.coreId,
                    step = payload.step.toLong(),
                    checkpointId = engine.trainer?.latestCheckpoint?.checkpointId,
                ),
            )
            is AxonEventPayload.CheckpointWritten -> AxonEvent.CheckpointWritten(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = CheckpointWrittenPayload(
                    checkpointId = payload.checkpointId,
                    moduleId = engine.trainer?.config?.moduleId ?: "",
                    candidateGenerationId = payload.candidateGenerationId,
                    step = payload.step.toLong(),
                    artifactRelpath = engine.trainer?.latestCheckpoint?.artifactRelpath,
                    artifactSha256 = engine.trainer?.latestCheckpoint?.artifactSha256 ?: "",
                    artifactBytes = engine.trainer?.latestCheckpoint?.artifactBytes,
                    optimizerIncluded = true,
                ),
            )
            is AxonEventPayload.HeldoutEvaluationCompleted -> AxonEvent.HeldoutEvaluationCompleted(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = HeldoutEvaluationCompletedPayload(
                    evaluationId = payload.evaluationId,
                    candidateGenerationId = payload.candidateGenerationId,
                    moduleId = engine.trainer?.config?.moduleId,
                    gatePassed = true,
                    metrics = payload.metrics,
                ),
            )
            is AxonEventPayload.ComputeWorkerConnected -> AxonEvent.ComputeWorkerConnected(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = ComputeWorkerConnectedPayload(
                    workerId = payload.workerId,
                    provider = payload.provider,
                    endpoint = "sim://workers/${payload.workerId}",
                    workerVersion = "sim-worker-1.0",
                ),
            )
            is AxonEventPayload.ComputeWorkerLost -> AxonEvent.ComputeWorkerLost(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = ComputeWorkerLostPayload(
                    workerId = payload.workerId,
                    provider = engine.workers.firstOrNull { it.workerId == payload.workerId }?.provider,
                ),
            )
            is AxonEventPayload.RecoveryCapsulePublished -> AxonEvent.RecoveryCapsulePublished(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = RecoveryCapsulePublishedPayload(
                    capsuleId = payload.capsuleId,
                    kind = CapsuleKind.FULL_ORGANISM,
                    archiveSha256 = payload.manifestSha256,
                    archiveBytes = capsulesById[payload.capsuleId]?.files?.values?.sumOf { it.size.toLong() } ?: 0L,
                    memberCount = payload.members.size,
                    gitCommit = "sim-local",
                ),
            )
            is AxonEventPayload.AgentMessageReceived -> AxonEvent.AgentMessageReceived(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = AgentMessageReceivedPayload(
                    messageId = "sim-msg-${event.seq}",
                    agentId = payload.agentId,
                    from = payload.agentId,
                    preview = payload.text.take(120),
                ),
            )
            is AxonEventPayload.AlertRaised -> AxonEvent.AlertRaised(
                eventId = event.eventId, seq = event.seq, timestamp = timestamp,
                payload = AlertRaisedPayload(
                    severity = when (payload.severity) {
                        "warning" -> AlertSeverity.WARNING
                        "critical" -> AlertSeverity.CRITICAL
                        else -> AlertSeverity.INFO
                    },
                    source = payload.source,
                    message = payload.message,
                ),
            )
            else -> null
        }
    }
}

/** Typed failure carrying the Control Plane error model. */
class ApiException(val error: ApiError) : Exception("${error.code}: ${error.message}")

/** Marker so UI layers can always tell sim provenance. */
const val SIM_CLIENT_SOURCE: String = SIMULATION_MARKER
