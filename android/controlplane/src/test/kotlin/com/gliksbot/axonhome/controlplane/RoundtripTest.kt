package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Roundtrip serialization for every model on the v1 surface. */
class RoundtripTest {

    private val json = Json { ignoreUnknownKeys = true }
    private val t = Instant.parse("2026-09-21T00:00:00Z")

    private inline fun <reified T> roundtrip(value: T): T {
        val encoded = json.encodeToString(value)
        val decoded = json.decodeFromString<T>(encoded)
        assertEquals(value, decoded, "roundtrip mismatch for ${value!!::class.simpleName}: $encoded")
        return decoded
    }

    @Test
    fun `common models roundtrip`() {
        roundtrip(ApiErrorEnvelope(error = ApiError(ErrorCode.SEQ_GAP, "gap", retryable = true, supportedVersions = listOf("v1"))))
        Freshness.entries.forEach { f ->
            assertEquals(f, json.decodeFromString("\"${f.name}\""))
        }
    }

    @Test
    fun `health models roundtrip`() {
        roundtrip(
            HealthResponse(
                freshness = Freshness.CACHED,
                observedAt = t,
                controlPlane = ControlPlaneStatus("degraded", "0.1.0", listOf("v1"), uptimeSeconds = 12),
                heart = HeartHealthSummary(
                    heartEpochId = "epoch-1",
                    heartbeatSequence = 10,
                    tickSequence = 9,
                    tickInFlight = true,
                    lastBeatOk = false,
                    lastFailure = "boom",
                ),
            )
        )
    }

    @Test
    fun `field models roundtrip`() {
        roundtrip(
            FieldHead(
                freshness = Freshness.LIVE,
                observedAt = t,
                head = BranchHead(generation = 5, fieldId = "f1", tickId = 42, parentFieldId = "f0"),
                health = HeartHealthSummary(
                    heartEpochId = "e", heartbeatSequence = 1, tickSequence = 42,
                    tickInFlight = false, lastBeatOk = true,
                ),
            )
        )
        roundtrip(
            FieldSnapshotView(
                freshness = Freshness.STALE,
                observedAt = t,
                fieldId = "f1",
                tickId = 42,
                regions = LogicalRegion.entries.map {
                    RegionSummary(it, 1, 10, RegionVisibility.ATTENDED, RegionWritePolicy.SEALED, "ab")
                },
                canonicalSha256 = "cd",
            )
        )
        roundtrip(
            TickList(
                freshness = Freshness.DISCONNECTED,
                observedAt = t,
                ticks = listOf(
                    TickRecord(
                        event = BranchEventKind.COMMIT, generation = 5, tickId = 42,
                        baseFieldId = "f0", successorFieldId = "f1",
                        deltaIds = listOf("d1"), heartCommitId = "hc1",
                        consolidatorCoreId = "core-alpha", timestamp = t,
                    )
                ),
            )
        )
    }

    @Test
    fun `masks models roundtrip`() {
        val policies = LogicalRegion.entries.associateWith {
            if (it == LogicalRegion.IDENTITY) MaskPolicy(MaskKind.ALL)
            else MaskPolicy(MaskKind.TAIL_PERCENT, tailPercent = 25)
        }
        val state = MaskState(
            freshness = Freshness.LIVE, observedAt = t,
            revision = 8, stateId = "ms1", viewId = "v1", policies = policies,
        )
        roundtrip(state)
        roundtrip(PutMasksRequest(policies = policies))
        roundtrip(PutMasksRequest(policies = policies, confirmToken = "tok", confirmed = true))
        roundtrip(PutMasksResponse(freshness = Freshness.LIVE, observedAt = t, applied = false, confirmToken = "tok", proposed = state))
    }

    @Test
    fun `cores and souls models roundtrip`() {
        roundtrip(
            CoreList(
                freshness = Freshness.LIVE, observedAt = t,
                cores = listOf(
                    CoreDescriptor(
                        coreId = "core-alpha", status = CoreStatus.ACTIVE,
                        architectureId = "living-d64-english-abc",
                        parameterGeneration = "english-candidate-x",
                        writableRegions = listOf(LogicalRegion.SCRATCH, LogicalRegion.RESPONSE_DRAFT),
                        soulId = "s1", soulGeneration = 7,
                    )
                ),
            )
        )
        roundtrip(
            CoreDetail(
                freshness = Freshness.LIVE, observedAt = t,
                core = CoreDescriptor(
                    coreId = "core-alpha", status = CoreStatus.OFFLINE_TRAINING,
                    architectureId = "untrained-reasoning-core-v1", parameterGeneration = "untrained",
                ),
                anatomy = CoreAnatomy(nHeads = 1, nLayers = 2, ffnDim = 131072, stateTokens = 4, pageSize = 32),
            )
        )
        roundtrip(
            SoulSummary(
                freshness = Freshness.LIVE, observedAt = t,
                coreId = "core-alpha", branchId = "live",
                head = SoulHead(soulId = "s2", generation = 8, parentSoulId = "s1"),
                layers = SoulTemperature.entries.map {
                    SoulLayerMeta(it, "application/x-axon-d64-recurrent-state", 1024, "ff")
                },
                receiptsTail = listOf(
                    SoulCommitReceiptMeta(
                        receiptId = "r1", beforeSoulId = "s1", afterSoulId = "s2",
                        generation = 8, phase = SoulPhase.CONSOLIDATED,
                        commitBinding = "canonical-field:f1",
                    )
                ),
            )
        )
    }

    @Test
    fun `trainer models roundtrip`() {
        assertEquals(13, TrainerCommandKind.entries.size)
        roundtrip(
            TrainerStatus(
                freshness = Freshness.LIVE, observedAt = t,
                inspection = TrainerInspection(
                    writerLeasePresent = true, leaseOwner = "trainer-host",
                    activeGenerations = listOf(
                        ActiveGeneration(
                            moduleId = "r64-english-reasoning",
                            candidateGenerationId = "english-candidate-x",
                            lifecycleStatus = TrainerLifecycleStatus.RUNNING,
                            currentStep = 24, latestLoss = 0.42, learningRate = 1e-4,
                        )
                    ),
                    latestCheckpoints = listOf(
                        CheckpointRecordMeta(
                            checkpointId = "ck1", moduleId = "m", candidateGenerationId = "g",
                            step = 24, artifactSha256 = "ab", optimizerIncluded = true,
                        )
                    ),
                    availableCommands = mapOf(TrainerCommandKind.STATUS to true, TrainerCommandKind.PAUSE to false),
                ),
            )
        )
        roundtrip(
            TrainerCommandRequest(
                kind = TrainerCommandKind.PREFLIGHT,
                args = JsonObject(emptyMap()),
                requestedBy = "jeff",
                idempotencyKey = "9c4f1e2a-7b3d",
            )
        )
        roundtrip(
            TrainerCommandResponse(
                freshness = Freshness.LIVE, observedAt = t,
                commandId = "cmd-1", kind = TrainerCommandKind.STATUS,
                status = TrainerCommandStatus.OK, message = "ok",
                auditId = "aud-1",
            )
        )
    }

    @Test
    fun `compute storage capsules audit agents stop models roundtrip`() {
        roundtrip(
            WorkerList(
                freshness = Freshness.LIVE, observedAt = t,
                workers = listOf(
                    Worker(
                        workerId = "w1", provider = "kaggle", endpoint = "kaggle://kernels/x",
                        status = WorkerStatus.BUSY, capabilities = CapabilitySet(supportsStop = true, supportsGpuMetrics = true),
                        accelerator = "nvidia-t4", lastHeartbeatAt = t, registeredAt = t,
                    )
                ),
            )
        )
        roundtrip(RegisterWorkerRequest(provider = "ssh", endpoint = "ssh://host", capabilities = CapabilitySet(supportsLaunch = true)))
        roundtrip(
            WorkerStopResponse(
                freshness = Freshness.LIVE, observedAt = t, workerId = "w1",
                ack = ComponentAck("w1", ComponentKind.COMPUTE_WORKER, ComponentAckStatus.CONFIRMED_STOPPED),
            )
        )
        roundtrip(
            ArtifactList(
                freshness = Freshness.CACHED, observedAt = t,
                artifacts = listOf(
                    Artifact(
                        artifactId = "a1", kind = ArtifactKind.CHECKPOINT, provider = "drive",
                        path = "checkpoints/ab.pt", sha256 = "ab", bytes = 100,
                        createdAt = t, contentAddressed = true, verifiedAt = t,
                    )
                ),
            )
        )
        roundtrip(RegisterArtifactRequest(kind = ArtifactKind.CLOUD_BUNDLE, provider = "sftp", path = "/b.tar.gz", sha256 = "cd", bytes = 1))
        roundtrip(
            CapsuleManifest(
                capsuleId = "cap-1", kind = CapsuleKind.FULL_ORGANISM,
                archiveSha256 = "aa", archiveBytes = 10,
                members = mapOf("m.json" to ManifestMember("bb", 10)),
                gitCommit = "843bbedd25347d367da1f979488a6ecb77128ab7", createdAt = t,
            )
        )
        roundtrip(
            CapsuleList(
                freshness = Freshness.LIVE, observedAt = t,
                capsules = listOf(CapsuleSummary("cap-1", CapsuleKind.CODE, "aa", createdAt = t, verificationStatus = CapsuleVerificationStatus.UNVERIFIED)),
            )
        )
        roundtrip(CapsuleExportRequest(kind = CapsuleKind.CHECKPOINT, moduleId = "m"))
        roundtrip(
            CapsuleVerificationReport(
                capsuleId = "cap-1", ok = false,
                checks = listOf(CapsuleCheck(CapsuleCheckName.MEMBER_SHA256, false, "mismatch")),
                mismatches = listOf("m.json"), quarantineDir = "jobs/j1/quarantine", verifiedAt = t,
            )
        )
        roundtrip(
            AuditList(
                freshness = Freshness.LIVE, observedAt = t,
                entries = listOf(
                    AuditEntry(
                        auditId = "aud-1", who = "jeff", device = "pixel",
                        command = "PUT /v1/masks", target = "conversation_history",
                        timestamp = t, previousState = "all", requestedState = "tail_percent:50",
                        result = AuditResult.SUCCESS,
                    )
                ),
            )
        )
        roundtrip(
            AgentList(
                freshness = Freshness.LIVE, observedAt = t,
                agents = listOf(
                    AgentRecord(
                        agentId = "ag-1", name = "kimmy", provider = "openai-compatible",
                        model = "k3", online = true, capabilities = AgentCapabilities(supportsStreaming = true),
                        vaultKeyRef = "vault://agents/kimmy",
                    )
                ),
            )
        )
        roundtrip(RegisterAgentRequest(name = "codex", provider = "generic-rest"))
        roundtrip(AgentMessage(messageId = "m1", agentId = "ag-1", from = "jeff", body = "status?", sentAt = t))
        roundtrip(AgentMessageRequest(body = "status?"))
        roundtrip(StopRequest(scope = StopScope.EMERGENCY))
        roundtrip(
            StopResponse(
                freshness = Freshness.LIVE, observedAt = t, scope = StopScope.PAUSE_AXON,
                allStopped = true,
                components = listOf(ComponentAck("h1", ComponentKind.HEART_HOST, ComponentAckStatus.CONFIRMED_STOPPED)),
                auditId = "aud-2",
            )
        )
    }

    @Test
    fun `event envelope roundtrips for all 18 types`() {
        val base = """"event_id":"evt-1","seq":1,"timestamp":"2026-09-21T00:00:00Z""""
        val events = listOf(
            """{"schema":"axon-home-event-v1",$base,"type":"TickStarted","payload":{"schema":"axon-heart-tick-identity-v1","tick_sequence":1,"heartbeat_id":"hb-1","base_field_id":"f0"}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"CoreFirstReturned","payload":{"schema":"axon-english-proposal-v1","tick_sequence":1,"core_id":"c1","proposal_id":"p1","participant_state":"returned"}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"FirstBarrierComplete","payload":{"schema":"axon-heart-english-proposal-workspace-v2","tick_sequence":1,"workspace_id":"ws1","pass":"first","returned_core_ids":["c1"]}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"CoreRefinedReturned","payload":{"schema":"axon-english-proposal-v1","tick_sequence":1,"core_id":"c1","proposal_id":"p2","participant_state":"returned"}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"RefinedBarrierComplete","payload":{"schema":"axon-heart-english-proposal-workspace-v2","tick_sequence":1,"workspace_id":"ws2","pass":"refined","returned_core_ids":["c1"]}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"ConsolidatorSelected","payload":{"tick_sequence":1,"core_id":"c1","participant_count":4}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"FinalReturned","payload":{"schema":"axon-tagged-final-verdict-v2","tick_sequence":1,"consolidator_core_id":"c1","verdict_id":"v1","region_tags":["#responseDraft#"]}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"HeartValidationStarted","payload":{"schema":"axon-heart-frozen-tick-image-v2","tick_sequence":1,"frozen_image_id":"fi1","view_id":"vw1"}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"FieldCommitted","payload":{"schema":"axon-heart-commit-v1","tick_sequence":1,"base_field_id":"f0","successor_field_id":"f1","delta_ids":["d1"],"commit_id":"hc1"}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"SoulTransitionAccepted","payload":{"schema":"axon-private-soul-commit-receipt-v1","core_id":"c1","receipt_id":"r1","before_soul_id":"s1","after_soul_id":"s2","generation":2,"phase":"consolidated"}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"TrainerStepAccepted","payload":{"schema":"axon-accepted-reasoning-training-step-v1","bundle_id":"b1","pointer_id":"pt1","module_id":"m1","candidate_generation_id":"g1","step":24}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"CheckpointWritten","payload":{"schema":"axon-trainer-candidate-checkpoint-v2","checkpoint_id":"ck1","module_id":"m1","candidate_generation_id":"g1","step":24,"artifact_sha256":"ab"}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"HeldoutEvaluationCompleted","payload":{"evaluation_id":"ev1","candidate_generation_id":"g1","gate_passed":true,"metrics":{"text_exact_rate":1.0}}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"ComputeWorkerConnected","payload":{"worker_id":"w1","provider":"ssh","capabilities":{"supportsStop":true}}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"ComputeWorkerLost","payload":{"worker_id":"w1","active_job_id":"j1"}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"RecoveryCapsulePublished","payload":{"schema":"axon-cloud-bundle-manifest-v1","capsule_id":"cap-1","kind":"full_organism","archive_sha256":"aa","archive_bytes":10,"member_count":8}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"AgentMessageReceived","payload":{"message_id":"m1","agent_id":"ag-1","from":"jeff"}}""",
            """{"schema":"axon-home-event-v1",$base,"type":"AlertRaised","payload":{"severity":"critical","source":"heart","message":"tick failed","related_event_ids":["evt-0"]}}""",
        )
        assertEquals(18, events.size, "contract defines exactly 18 event types (arch §6.2)")
        events.forEach { text ->
            val event = AxonEvent.json.decodeFromString<AxonEvent>(text)
            assertEquals(AxonEvent.ENVELOPE_SCHEMA, event.schema)
            assertEquals(1, event.seq)
            // Roundtrip back through the same discriminator-aware Json.
            val reencoded = AxonEvent.json.encodeToString(AxonEvent.serializer(), event)
            val decoded = AxonEvent.json.decodeFromString<AxonEvent>(reencoded)
            assertEquals(event, decoded)
            assertTrue(reencoded.contains("\"type\":\"${event.typeName}\""), "discriminator missing in $reencoded")
        }
    }
}
