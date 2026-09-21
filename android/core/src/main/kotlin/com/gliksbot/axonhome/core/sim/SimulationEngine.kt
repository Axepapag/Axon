package com.gliksbot.axonhome.core.sim

import com.gliksbot.axonhome.core.audit.AuditLog
import com.gliksbot.axonhome.core.capsule.RecoveryCapsule
import com.gliksbot.axonhome.core.capsule.RecoveryCapsuleBuilder
import com.gliksbot.axonhome.core.domain.AcceptedStepBundle
import com.gliksbot.axonhome.core.domain.AcceptedStepPointer
import com.gliksbot.axonhome.core.domain.AnatomyConfig
import com.gliksbot.axonhome.core.domain.AxonJson
import com.gliksbot.axonhome.core.domain.CORE_WRITABLE_REGIONS
import com.gliksbot.axonhome.core.domain.CandidateLifecycleStatus
import com.gliksbot.axonhome.core.domain.CheckpointRecord
import com.gliksbot.axonhome.core.domain.CoreDescriptor
import com.gliksbot.axonhome.core.domain.CoreStatus
import com.gliksbot.axonhome.core.domain.DeleteText
import com.gliksbot.axonhome.core.domain.EnglishProposal
import com.gliksbot.axonhome.core.domain.FieldDelta
import com.gliksbot.axonhome.core.domain.HeartHealth
import com.gliksbot.axonhome.core.domain.HeartRegionMaskState
import com.gliksbot.axonhome.core.domain.InsertText
import com.gliksbot.axonhome.core.domain.LearningPolicy
import com.gliksbot.axonhome.core.domain.LogicalRegion
import com.gliksbot.axonhome.core.domain.MirrorEnvelope
import com.gliksbot.axonhome.core.domain.OptimizationStepReceipt
import com.gliksbot.axonhome.core.domain.PrecisionMode
import com.gliksbot.axonhome.core.domain.ROLLING_RETENTION
import com.gliksbot.axonhome.core.domain.RegionMaskPolicy
import com.gliksbot.axonhome.core.domain.ResourceTranche
import com.gliksbot.axonhome.core.domain.SharedFieldSnapshot
import com.gliksbot.axonhome.core.domain.SoulCommitReceipt
import com.gliksbot.axonhome.core.domain.SoulSnapshot
import com.gliksbot.axonhome.core.domain.SoulTemperature
import com.gliksbot.axonhome.core.domain.SoulTransition
import com.gliksbot.axonhome.core.domain.StateFreshness
import com.gliksbot.axonhome.core.domain.TechnicalFinalVerdict
import com.gliksbot.axonhome.core.domain.TrancheContinuation
import com.gliksbot.axonhome.core.domain.applyDelta
import com.gliksbot.axonhome.core.domain.applySoulTransition
import com.gliksbot.axonhome.core.domain.canonicalSha256
import com.gliksbot.axonhome.core.domain.genesisSoul
import com.gliksbot.axonhome.core.domain.renderFinalVerdict
import com.gliksbot.axonhome.core.events.AxonEventPayload
import com.gliksbot.axonhome.core.events.EventLog
import com.gliksbot.axonhome.core.stop.ComponentAck
import com.gliksbot.axonhome.core.stop.EmergencyStopResult
import com.gliksbot.axonhome.core.stop.SafeStopResult
import com.gliksbot.axonhome.core.stop.StopKind
import com.gliksbot.axonhome.core.stop.StopResult
import kotlinx.datetime.Instant
import kotlinx.serialization.encodeToString
import kotlin.math.exp
import kotlin.random.Random

/** Marker embedded in every simulated payload so it can never masquerade as live. */
const val SIMULATION_MARKER = "SIMULATION"

/** Fixed simulation epoch so timestamps are deterministic given the seed. */
const val SIM_EPOCH_SECONDS = 1_770_000_000L

@kotlinx.serialization.Serializable
data class SimCore(
    val descriptor: CoreDescriptor,
    val anatomy: AnatomyConfig,
    val consolidatorEligible: Boolean,
)

@kotlinx.serialization.Serializable
data class SimWorker(
    @kotlinx.serialization.SerialName("worker_id") val workerId: String,
    val provider: String,
    val accelerator: String,
    val connected: Boolean,
)

@kotlinx.serialization.Serializable
data class SimTrainingConfig(
    @kotlinx.serialization.SerialName("module_id") val moduleId: String = "r64-english-reasoning",
    @kotlinx.serialization.SerialName("core_id") val coreId: String = "sim-core-1",
    @kotlinx.serialization.SerialName("tranche_steps") val trancheSteps: Int = 4,
    @kotlinx.serialization.SerialName("learning_rate") val learningRate: Double = 1e-4,
    @kotlinx.serialization.SerialName("objective_program_id") val objectiveProgramId: String =
        "teacher_forced_decode:sequence_cross_entropy:head0",
    @kotlinx.serialization.SerialName("eval_interval") val evalInterval: Int = 5,
)

/** Immutable record of one simulated tick, retained for the tick inspector. */
data class TickRecord(
    val tickNumber: Int,
    val baseFieldId: String,
    val firstProposals: List<EnglishProposal>,
    val refinedProposals: List<EnglishProposal>,
    val consolidatorCoreId: String,
    val verdict: TechnicalFinalVerdict?,
    val deltaIds: List<String>,
    val resultFieldId: String?,
    val rejected: Boolean,
    val rejectionReason: String? = null,
)

/** Mutable trainer state, deterministic given the engine seed. */
class SimTrainer internal constructor(
    val config: SimTrainingConfig,
    val policy: LearningPolicy,
    val anatomy: AnatomyConfig,
    val planId: String,
    val baseGenerationId: String,
    val candidateGenerationId: String,
    var candidateSoul: SoulSnapshot,
) {
    var lifecycle: CandidateLifecycleStatus = CandidateLifecycleStatus.RUNNING
        internal set
    var globalStep: Int = 0
        internal set
    var tranche: ResourceTranche? = null
        internal set
    val continuations = mutableListOf<TrancheContinuation>()
    val bundles = mutableListOf<AcceptedStepBundle>()
    val receipts = mutableListOf<OptimizationStepReceipt>()
    val soulReceipts = mutableListOf<SoulCommitReceipt>()
    /** Rolling window: only the newest ROLLING_RETENTION checkpoints are retained. */
    val retainedCheckpoints = mutableListOf<CheckpointRecord>()
    var pointer: AcceptedStepPointer? = null
        internal set
    val heldoutEvaluations = mutableListOf<Map<String, Double>>()
    var lastLoss: Double = Double.NaN
        internal set

    val latestCheckpoint: CheckpointRecord? get() = retainedCheckpoints.lastOrNull()
}

/**
 * Deterministic, seeded simulation of the Axon organism for Axon Home
 * development (mission §15). Pipeline shape mirrors the real Heart:
 *
 *   F_t → FIRST proposals (English) → FIRST board → REFINED → refined board
 *     → rotating consolidator participants[(tick-1) % n] → FINAL verdict
 *     → Heart parse/validate → FieldDelta → F_{t+1}
 *
 * plus per-pass soul transitions, trainer steps with rolling-3 checkpoints,
 * tranche exhaustion, compute workers, recovery capsules and Roundtable
 * chatter. Every emitted payload carries the SIMULATION marker. Given the
 * same seed and the same call sequence, all ids and hashes reproduce exactly.
 */
class SimulationEngine(
    val seed: Long = 20260919L,
    coreCount: Int = 4,
) {
    private val rng = Random(seed)
    val eventLog = EventLog()
    val auditLog = AuditLog()

    private var clockSeconds = SIM_EPOCH_SECONDS
    private fun now(): String = Instant.fromEpochSeconds(clockSeconds++).toString()

    private val epochId = "sim-epoch-" + canonicalSha256("epoch|$seed").take(12)
    private var heartbeatSequence = 0L

    /** The canonical shared field (simulated). */
    var field: SharedFieldSnapshot
        private set

    var maskState: HeartRegionMaskState = HeartRegionMaskState.defaults()
        private set

    val cores: List<SimCore>
    private val souls = mutableMapOf<String, SoulSnapshot>()
    private val soulReceiptChains = mutableMapOf<String, MutableList<SoulCommitReceipt>>()

    val workers = mutableListOf<SimWorker>()
    val agents = listOf("codex", "kimi", "chatgpt", "claude")

    var trainer: SimTrainer? = null
        private set

    var organismPaused: Boolean = false
        private set

    val tickHistory = mutableListOf<TickRecord>()

    private val proposalVocab = listOf(
        "coherence", "signal", "region", "attention", "evidence", "delta",
        "barrier", "lineage", "receipt", "rail", "provenance", "boundary",
    )

    init {
        require(coreCount >= 1) { "simulation needs at least one core" }
        // Four heterogeneous D64 cores; the last one is consolidator-ineligible.
        val anatomies = listOf(
            AnatomyConfig(nHeads = 1, nLayers = 2, ffnDim = 131072, stateTokens = 4, pageSize = 32),
            AnatomyConfig(nHeads = 2, nLayers = 3, ffnDim = 65536, stateTokens = 8, pageSize = 32),
            AnatomyConfig(nHeads = 4, nLayers = 2, ffnDim = 262144, stateTokens = 4, pageSize = 64),
            AnatomyConfig(nHeads = 1, nLayers = 4, ffnDim = 32768, stateTokens = 16, pageSize = 32),
        )
        cores = (0 until coreCount).map { index ->
            val anatomy = anatomies[index % anatomies.size]
            SimCore(
                descriptor = CoreDescriptor(
                    coreId = "sim-core-${index + 1}",
                    status = CoreStatus.ACTIVE,
                    writableRegions = CORE_WRITABLE_REGIONS.toList(),
                    architectureId = anatomy.architectureId,
                    parameterGeneration = "untrained",
                ),
                anatomy = anatomy,
                consolidatorEligible = index < coreCount - 1, // last core never consolidates
            )
        }
        for (core in cores) {
            souls[core.descriptor.coreId] = genesisSoul(
                core.descriptor.coreId, core.anatomy.architectureId, "untrained", "seed:$seed",
            )
            soulReceiptChains[core.descriptor.coreId] = mutableListOf()
        }

        field = SharedFieldSnapshot.fromTexts(
            texts = mapOf(
                LogicalRegion.IDENTITY to
                    "AXON HOME SIMULATION ORGANISM — synthetic state, NOT a live organism [$SIMULATION_MARKER]",
                LogicalRegion.TASK_STATE to "idle [$SIMULATION_MARKER]",
                LogicalRegion.SITUATION_AWARENESS to "no live host attached [$SIMULATION_MARKER]",
            ),
            tickId = 0,
            source = SIMULATION_MARKER,
            provenance = "simulation:genesis:$seed",
        )

        workers += SimWorker("sim-worker-kaggle-01", "kaggle", "T4 x2", connected = true)
        workers += SimWorker("sim-worker-ssh-01", "ssh", "none", connected = true)
        emit(AxonEventPayload.ComputeWorkerConnected("sim-worker-kaggle-01", "kaggle"))
        emit(AxonEventPayload.ComputeWorkerConnected("sim-worker-ssh-01", "ssh"))
    }

    // ------------------------------------------------------------------
    // Event helper
    // ------------------------------------------------------------------

    private fun emit(payload: AxonEventPayload) = eventLog.append(payload, now())

    // ------------------------------------------------------------------
    // Organism observation
    // ------------------------------------------------------------------

    val activeCores: List<SimCore> get() = cores.filter { it.descriptor.status == CoreStatus.ACTIVE }
    private val consolidatorPool: List<SimCore> get() = activeCores.filter { it.consolidatorEligible }

    fun soulOf(coreId: String): SoulSnapshot = souls.getValue(coreId)

    fun soulReceiptsOf(coreId: String): List<SoulCommitReceipt> =
        soulReceiptChains[coreId]?.toList() ?: emptyList()

    fun health(): HeartHealth = HeartHealth(
        epochId = epochId,
        heartbeatSequence = heartbeatSequence,
        tickSequence = field.tickId.toLong(),
        headFieldId = field.fieldId,
        headTickId = field.tickId,
        tickInFlight = false,
        maskRevision = maskState.revision,
        maskStateId = maskState.stateId,
        lastViewId = deriveViewId(),
        lastFailure = null,
    )

    private fun deriveViewId(): String =
        "view-" + canonicalSha256(maskState.policies.toSortedMap().mapValues { it.value.toCanonicalDict() }).take(24)

    /** Screen-facing mirror; ALWAYS freshness-labelled and SIMULATION-sourced. */
    fun fieldMirror(freshness: StateFreshness = StateFreshness.LIVE): MirrorEnvelope<SharedFieldSnapshot> =
        MirrorEnvelope(field, freshness, now(), SIMULATION_MARKER)

    fun healthMirror(freshness: StateFreshness = StateFreshness.LIVE): MirrorEnvelope<HeartHealth> =
        MirrorEnvelope(health(), freshness, now(), SIMULATION_MARKER)

    // ------------------------------------------------------------------
    // Ingress + tick pipeline
    // ------------------------------------------------------------------

    /** Durable ingress: insert user text into user_input via a Heart valve delta. */
    fun submitUserInput(text: String): FieldDelta {
        require(text.isNotBlank()) { "user input must be non-blank" }
        val base = field
        val region = base.region(LogicalRegion.USER_INPUT)
        val delta = FieldDelta(
            baseFieldId = base.fieldId,
            baseTickId = base.tickId,
            authorCoreId = "heart-valve:user_ingress",
            passId = "heart_ingress",
            operations = listOf(
                InsertText(
                    region = LogicalRegion.USER_INPUT,
                    offset = region.text.length,
                    text = text.trim() + "\n",
                    provenance = "simulation:ingress:$SIMULATION_MARKER",
                )
            ),
            evidence = listOf("simulation-ingress"),
        )
        field = applyDelta(base, delta, permittedRegions = setOf(LogicalRegion.USER_INPUT))
        auditLog.record(
            who = "operator", device = "sim", command = "submit_user_input",
            target = "user_input", timestamp = now(),
            previousState = base.fieldId, requestedState = field.fieldId,
            result = "committed",
        )
        return delta
    }

    private fun synthProposal(core: SimCore, base: SharedFieldSnapshot, passId: String, tickNumber: Int): EnglishProposal {
        val words = List(4) { proposalVocab[rng.nextInt(proposalVocab.size)] }
        val userSnippet = base.region(LogicalRegion.USER_INPUT).text.trim().take(48)
        val boardNote = if (passId == "refined") " after reading the FIRST board" else ""
        return EnglishProposal(
            baseFieldId = base.fieldId,
            baseTickId = base.tickId,
            authorCoreId = core.descriptor.coreId,
            passId = passId,
            railDModel = 64,
            text = "[$SIMULATION_MARKER] $passId proposal by ${core.descriptor.coreId} " +
                "at tick $tickNumber$boardNote: ${words.joinToString(" ")}" +
                if (userSnippet.isNotEmpty()) "; heard '$userSnippet'" else "",
            evidence = listOf("simulation"),
        )
    }

    private fun soulPass(coreId: String, phase: String, tickUid: String, commitBinding: String? = null): SoulCommitReceipt {
        val before = souls.getValue(coreId)
        val transition = SoulTransition(
            coreId = coreId,
            branchId = "live",
            beforeSoulId = before.soulId,
            phase = phase,
            updatedTemperatures = listOf(SoulTemperature.HOT),
            tickUid = tickUid,
            requestId = "sim-req-" + canonicalSha256("$tickUid|$coreId|$phase|${before.soulId}").take(16),
        )
        val (after, receipt) = applySoulTransition(before, transition, commitBinding)
        souls[coreId] = after
        soulReceiptChains.getValue(coreId).add(receipt)
        emit(
            AxonEventPayload.SoulTransitionAccepted(
                coreId = coreId, soulId = after.soulId, generation = after.generation,
                receiptId = receipt.receiptId, phase = phase,
            )
        )
        return receipt
    }

    /**
     * Run one full heartbeat: FIRST → REFINED → rotating consolidator → FINAL
     * → Heart validate → commit → F_{t+1}. Returns the tick record, or null
     * if the organism is paused. Emits events in the exact mission order.
     */
    fun tick(): TickRecord? {
        if (organismPaused) {
            emit(AxonEventPayload.AlertRaised("info", "tick suppressed: organism paused", SIMULATION_MARKER))
            return null
        }
        heartbeatSequence++
        val base = field
        val tickNumber = base.tickId + 1
        val tickUid = "sim-tick-$tickNumber-" + canonicalSha256("$seed|$tickNumber|${base.fieldId}").take(12)
        // Failure dice are rolled up-front so every tick consumes the same
        // amount of randomness regardless of branch outcome (determinism).
        val validationReject = rng.nextInt(24) == 0
        val workerEventRoll = rng.nextInt(40)

        emit(AxonEventPayload.TickStarted(tickNumber, base.fieldId))

        // FIRST pass — every active core returns an English proposal; each
        // pass commits a soul transition.
        val participants = activeCores
        val firstProposals = mutableListOf<EnglishProposal>()
        for (core in participants) {
            val proposal = synthProposal(core, base, "first", tickNumber)
            firstProposals += proposal
            emit(AxonEventPayload.CoreFirstReturned(tickNumber, core.descriptor.coreId, proposal.proposalId))
            soulPass(core.descriptor.coreId, "first", tickUid)
        }
        emit(AxonEventPayload.FirstBarrierComplete(tickNumber, participants.map { it.descriptor.coreId }))

        // REFINED pass — cores see the FIRST board.
        val refinedProposals = mutableListOf<EnglishProposal>()
        for (core in participants) {
            val proposal = synthProposal(core, base, "refined", tickNumber)
            refinedProposals += proposal
            emit(AxonEventPayload.CoreRefinedReturned(tickNumber, core.descriptor.coreId, proposal.proposalId))
            soulPass(core.descriptor.coreId, "refined", tickUid)
        }
        emit(AxonEventPayload.RefinedBarrierComplete(tickNumber, participants.map { it.descriptor.coreId }))

        // Rotating consolidator: participants[(tick-1) % n] over the eligible pool.
        val pool = consolidatorPool
        val consolidator = pool[(tickNumber - 1) % pool.size]
        emit(AxonEventPayload.ConsolidatorSelected(tickNumber, consolidator.descriptor.coreId))

        val userSnippet = base.region(LogicalRegion.USER_INPUT).text.trim().take(64)
        val verdict = TechnicalFinalVerdict(
            baseFieldId = base.fieldId,
            baseTickId = base.tickId,
            authorCoreId = consolidator.descriptor.coreId,
            railDModel = 64,
            text = renderFinalVerdict(
                listOf(
                    LogicalRegion.RESPONSE_DRAFT to (
                        "[$SIMULATION_MARKER] reply t$tickNumber by ${consolidator.descriptor.coreId}: " +
                            if (userSnippet.isNotEmpty()) "acknowledged '$userSnippet'" else "standing by"
                        ),
                    LogicalRegion.SCRATCH to
                        "[$SIMULATION_MARKER] scratch t$tickNumber: consolidated ${participants.size} proposals",
                )
            ),
            evidence = firstProposals.map { it.proposalId } + refinedProposals.map { it.proposalId },
        )
        emit(AxonEventPayload.FinalReturned(tickNumber, consolidator.descriptor.coreId, verdict.verdictId))
        emit(AxonEventPayload.HeartValidationStarted(tickNumber, verdict.verdictId))

        if (validationReject) {
            val reason = "heart validation rejected the verdict materialization " +
                "[$SIMULATION_MARKER seeded failure]"
            emit(AxonEventPayload.AlertRaised("warning", reason, SIMULATION_MARKER))
            val record = TickRecord(
                tickNumber, base.fieldId, firstProposals, refinedProposals,
                consolidator.descriptor.coreId, verdict, emptyList(), null,
                rejected = true, rejectionReason = reason,
            )
            tickHistory += record
            ambientEvents(tickNumber, workerEventRoll)
            return record
        }

        // Heart materializes FINAL into a typed delta and commits atomically.
        val delta = verdict.materialize(base)
        var next = applyDelta(base, delta)
        val deltaIds = mutableListOf(delta.deltaId)

        // Turn materialization under Heart authority: frame the completed turn
        // into conversation_history and clear user_input.
        val userText = base.region(LogicalRegion.USER_INPUT).text
        if (userText.isNotBlank()) {
            val history = next.region(LogicalRegion.CONVERSATION_HISTORY)
            val frame = "--- turn t$tickNumber [$SIMULATION_MARKER] ---\nuser: ${userText.trim()}\n" +
                "axon: ${next.region(LogicalRegion.RESPONSE_DRAFT).text.trim()}\n"
            val turnDelta = FieldDelta(
                baseFieldId = next.fieldId,
                baseTickId = next.tickId,
                authorCoreId = "heart",
                passId = "heart_turn_finalization",
                operations = listOf(
                    InsertText(
                        region = LogicalRegion.CONVERSATION_HISTORY,
                        offset = history.text.length,
                        text = frame,
                        provenance = "simulation:turn_finalization",
                    ),
                    DeleteText(
                        region = LogicalRegion.USER_INPUT,
                        startBound = 0,
                        endBound = next.region(LogicalRegion.USER_INPUT).text.length,
                        provenance = "simulation:turn_finalization",
                    ),
                ),
                evidence = listOf(delta.deltaId),
            )
            next = applyDelta(
                next, turnDelta,
                permittedRegions = setOf(LogicalRegion.CONVERSATION_HISTORY, LogicalRegion.USER_INPUT),
            )
            deltaIds += turnDelta.deltaId
        }

        // Consolidator soul finalized, bound to the canonical successor field.
        soulPass(
            consolidator.descriptor.coreId, "consolidated", tickUid,
            commitBinding = "canonical-field:${next.fieldId}",
        )

        field = next
        emit(
            AxonEventPayload.FieldCommitted(
                tickId = tickNumber, fieldId = next.fieldId,
                parentFieldId = base.fieldId, deltaIds = deltaIds,
            )
        )

        val record = TickRecord(
            tickNumber, base.fieldId, firstProposals, refinedProposals,
            consolidator.descriptor.coreId, verdict, deltaIds, next.fieldId,
            rejected = false,
        )
        tickHistory += record
        ambientEvents(tickNumber, workerEventRoll)
        return record
    }

    private fun ambientEvents(tickNumber: Int, workerEventRoll: Int) {
        // Seeded, deterministic ambient failures: worker lost / recovered.
        when (workerEventRoll) {
            0 -> {
                val lost = workers.filter { it.connected }.randomOrNull(rng) ?: return
                replaceWorker(lost.copy(connected = false))
                emit(AxonEventPayload.ComputeWorkerLost(lost.workerId, "heartbeat timeout [$SIMULATION_MARKER]"))
            }
            1 -> {
                val back = workers.filter { !it.connected }.randomOrNull(rng) ?: return
                replaceWorker(back.copy(connected = true))
                emit(AxonEventPayload.ComputeWorkerConnected(back.workerId, back.provider))
            }
            2 -> emit(
                AxonEventPayload.AlertRaised(
                    "info", "stale delta rejected at boundary [$SIMULATION_MARKER seeded]", SIMULATION_MARKER,
                )
            )
        }
        if (tickNumber % 6 == 0) {
            val agent = agents[(tickNumber / 6 - 1) % agents.size]
            emit(
                AxonEventPayload.AgentMessageReceived(
                    agent, "roundtable",
                    "[$SIMULATION_MARKER] $agent reviewed tick $tickNumber; field lineage intact.",
                )
            )
        }
    }

    private fun replaceWorker(worker: SimWorker) {
        val index = workers.indexOfFirst { it.workerId == worker.workerId }
        if (index >= 0) workers[index] = worker else workers += worker
    }

    fun connectWorker(workerId: String, provider: String, accelerator: String = "unknown") {
        replaceWorker(SimWorker(workerId, provider, accelerator, connected = true))
        emit(AxonEventPayload.ComputeWorkerConnected(workerId, provider))
    }

    fun loseWorker(workerId: String, reason: String) {
        val worker = workers.firstOrNull { it.workerId == workerId } ?: return
        replaceWorker(worker.copy(connected = false))
        emit(AxonEventPayload.ComputeWorkerLost(workerId, "$reason [$SIMULATION_MARKER]"))
    }

    fun receiveAgentMessage(agentId: String, text: String, channel: String = "roundtable") {
        emit(AxonEventPayload.AgentMessageReceived(agentId, channel, "[$SIMULATION_MARKER] $text"))
    }

    // ------------------------------------------------------------------
    // Mask / attention control (derived view only — never canonical state)
    // ------------------------------------------------------------------

    fun setRegionMaskPolicy(region: LogicalRegion, policy: RegionMaskPolicy): HeartRegionMaskState {
        val before = maskState
        maskState = maskState.withPolicy(region, policy)
        auditLog.record(
            who = "operator", device = "sim", command = "set_region_mask_policy",
            target = region.name, timestamp = now(),
            previousState = "rev ${before.revision}/${before.stateId.take(12)}",
            requestedState = "rev ${maskState.revision}/${maskState.stateId.take(12)}",
            result = "view-updated",
        )
        return maskState
    }

    // ------------------------------------------------------------------
    // Trainer simulation
    // ------------------------------------------------------------------

    fun startTraining(config: SimTrainingConfig = SimTrainingConfig()): SimTrainer {
        require(trainer == null || trainer!!.lifecycle in setOf(
            CandidateLifecycleStatus.COMPLETED,
            CandidateLifecycleStatus.REJECTED,
            CandidateLifecycleStatus.RETIRED,
        )) { "a training candidate is already active" }
        val core = cores.first { it.descriptor.coreId == config.coreId }
        val policy = LearningPolicy(
            learningRate = config.learningRate,
            maxGradientL2 = 10.0,
            maxUpdateL2 = 1.0,
            precision = PrecisionMode.FP32,
            objectiveProgramId = config.objectiveProgramId,
        )
        val baseGenerationId = "english-init-" +
            canonicalSha256("${core.anatomy.architectureId}|$seed|1.5").take(20)
        val candidateGenerationId = "english-candidate-" + canonicalSha256(
            "${config.moduleId}|$baseGenerationId|${core.anatomy.architectureId}|" +
                "sim-curriculum|${policy.policyId}|reasoning|8"
        ).take(20)
        val planId = "plan-" + canonicalSha256(
            "$baseGenerationId|$candidateGenerationId|${config.moduleId}"
        ).take(20)
        val state = SimTrainer(
            config = config,
            policy = policy,
            anatomy = core.anatomy,
            planId = planId,
            baseGenerationId = baseGenerationId,
            candidateGenerationId = candidateGenerationId,
            candidateSoul = genesisSoul(
                config.coreId, core.anatomy.architectureId, candidateGenerationId,
                "candidate:$seed:$candidateGenerationId",
            ),
        )
        state.tranche = ResourceTranche(
            moduleId = config.moduleId,
            candidateGenerationId = candidateGenerationId,
            planId = planId,
            learningPolicyId = policy.policyId,
            baseGlobalStep = 0,
            steps = config.trancheSteps,
            parentBundleId = null,
            purpose = "initial tranche [$SIMULATION_MARKER]",
        )
        trainer = state
        auditLog.record(
            who = "operator", device = "sim", command = "start_training",
            target = candidateGenerationId, timestamp = now(),
            previousState = "none", requestedState = "running", result = "running",
        )
        return state
    }

    /**
     * Attempt one accepted global step. Returns false when the tranche does
     * not admit the next step: the candidate checkpoints and pauses for a
     * later tranche (real trainer doctrine).
     */
    fun stepTraining(): Boolean {
        val state = trainer ?: return false
        if (state.lifecycle != CandidateLifecycleStatus.RUNNING) return false
        val tranche = state.tranche ?: return false
        if (!tranche.admits(state.globalStep)) {
            state.lifecycle = CandidateLifecycleStatus.PAUSED
            emit(
                AxonEventPayload.AlertRaised(
                    "info",
                    "resource tranche does not admit the next optimizer step; " +
                        "checkpoint and pause for a later tranche [$SIMULATION_MARKER]",
                    SIMULATION_MARKER,
                )
            )
            return false
        }

        val step = state.globalStep + 1
        val noise = (rng.nextDouble() - 0.5) * 0.04
        val loss = 2.5 * exp(-0.15 * step) + noise
        val gradL2 = 0.8 * exp(-0.1 * step) + rng.nextDouble() * 0.1
        val receipt = OptimizationStepReceipt(
            step = step,
            microStep = step,
            loss = loss,
            gradientL2 = gradL2,
            gradientClipNorm = state.policy.gradientClipNorm,
            learningRate = state.policy.learningRate,
            weightDecay = state.policy.weightDecay,
            precisionMode = state.policy.precision,
            updateL2 = gradL2 * state.policy.learningRate,
            telemetryFrameId = "sim-telemetry-" + canonicalSha256("$step|$loss|$seed").take(16),
            changedTensorNames = listOf("layers.0.attn", "layers.0.ffn"),
            unchangedTensorNames = listOf("substrate"),
        )
        state.receipts += receipt

        // Accepted-step unroll: exactly 3 soul transitions per accepted step.
        val soulReceiptIds = mutableListOf<String>()
        repeat(3) { leg ->
            val before = state.candidateSoul
            val transition = SoulTransition(
                coreId = state.config.coreId,
                branchId = "candidate",
                beforeSoulId = before.soulId,
                phase = "training",
                updatedTemperatures = listOf(SoulTemperature.HOT),
                requestId = "sim-train-$step-$leg",
            )
            val (after, soulReceipt) = applySoulTransition(before, transition)
            state.candidateSoul = after
            state.soulReceipts += soulReceipt
            soulReceiptIds += soulReceipt.receiptId
        }

        val artifactSha = canonicalSha256("artifact|${state.candidateGenerationId}|$step|$seed")
        val checkpoint = CheckpointRecord(
            moduleId = state.config.moduleId,
            baseGenerationId = state.baseGenerationId,
            candidateGenerationId = state.candidateGenerationId,
            planId = state.planId,
            authorizationId = "sim-authorization-" + canonicalSha256(state.planId).take(16),
            learningPolicyId = state.policy.policyId,
            step = step,
            microStep = step,
            accumulationIndex = 0,
            parameterManifestId = "sim-params-" + canonicalSha256("${state.anatomy.architectureId}|$step").take(16),
            artifactRelpath = "checkpoints/$artifactSha.pt",
            artifactSha256 = artifactSha,
            artifactBytes = 1_048_576L + (step * 4096L),
            optimizerIncluded = true,
            gradientStateIncluded = false,
            scalerIncluded = false,
            currentLearningRate = state.policy.learningRate,
            accumulatedLossSum = loss,
            previousCheckpointId = state.latestCheckpoint?.checkpointId,
        )
        // Rolling-3 retention: only the newest ROLLING_RETENTION checkpoints live.
        state.retainedCheckpoints += checkpoint
        while (state.retainedCheckpoints.size > ROLLING_RETENTION) {
            state.retainedCheckpoints.removeAt(0)
        }

        val bundle = AcceptedStepBundle(
            intentId = "sim-intent-" + canonicalSha256("$step|${receipt.receiptId}").take(16),
            moduleId = state.config.moduleId,
            candidateGenerationId = state.candidateGenerationId,
            coreId = state.config.coreId,
            step = step,
            optimizationReceiptId = receipt.receiptId,
            checkpointId = checkpoint.checkpointId,
            candidateSoulManifestId = "sim-soul-manifest-" + state.candidateSoul.soulId.take(16),
            beforeSoulId = state.candidateSoul.parentSoulId ?: state.candidateSoul.soulId,
            afterSoulId = state.candidateSoul.soulId,
            soulReceiptIds = soulReceiptIds,
            previousBundleId = state.bundles.lastOrNull()?.bundleId,
        )
        state.bundles += bundle
        state.pointer = AcceptedStepPointer(
            moduleId = state.config.moduleId,
            candidateGenerationId = state.candidateGenerationId,
            currentBundleId = bundle.bundleId,
            currentStep = step,
            rollingBundleIds = state.bundles.takeLast(ROLLING_RETENTION).map { it.bundleId },
        )
        state.globalStep = step
        state.lastLoss = loss

        emit(
            AxonEventPayload.TrainerStepAccepted(
                moduleId = state.config.moduleId,
                candidateGenerationId = state.candidateGenerationId,
                step = step, loss = loss, gradientL2 = gradL2, bundleId = bundle.bundleId,
            )
        )
        emit(
            AxonEventPayload.CheckpointWritten(
                checkpointId = checkpoint.checkpointId, step = step,
                candidateGenerationId = state.candidateGenerationId,
            )
        )
        if (step % state.config.evalInterval == 0) {
            val metrics = mapOf(
                "text_exact_rate" to minOf(1.0, 0.2 + 0.05 * step),
                "text_teacher_forced_content_accuracy" to minOf(1.0, 0.4 + 0.06 * step),
                "text_teacher_forced_eos_accuracy" to minOf(1.0, 0.5 + 0.05 * step),
                "complete_field_coverage_rate" to 1.0,
                "final_verdict_valid_rate" to minOf(1.0, 0.6 + 0.04 * step),
            )
            state.heldoutEvaluations += metrics
            emit(
                AxonEventPayload.HeldoutEvaluationCompleted(
                    evaluationId = "sim-eval-" + canonicalSha256("$step|$seed").take(16),
                    candidateGenerationId = state.candidateGenerationId,
                    metrics = metrics,
                )
            )
        }
        return true
    }

    /** Pause at tranche exhaustion → resume under a new tranche (continuation). */
    fun requestNextTranche(steps: Int? = null): ResourceTranche {
        val state = trainer ?: throw IllegalStateException("no training candidate")
        require(state.lifecycle == CandidateLifecycleStatus.PAUSED) {
            "continuation requires a paused candidate"
        }
        val lastBundle = state.bundles.lastOrNull()
            ?: throw IllegalStateException("no accepted bundle to continue from")
        val lastCheckpoint = state.latestCheckpoint
            ?: throw IllegalStateException("no checkpoint to continue from")
        val priorTranche = state.tranche!!
        val continuation = TrancheContinuation(
            parentBundleId = lastBundle.bundleId,
            parentCheckpointId = lastCheckpoint.checkpointId,
            parentOptimizerReceiptId = state.receipts.last().receiptId,
            parentSoulId = state.candidateSoul.soulId,
            parentGlobalStep = state.globalStep,
            priorTrancheId = priorTranche.trancheId,
        )
        state.continuations += continuation
        val next = ResourceTranche(
            moduleId = state.config.moduleId,
            candidateGenerationId = state.candidateGenerationId,
            planId = state.planId,
            learningPolicyId = state.policy.policyId,
            baseGlobalStep = state.globalStep,
            steps = steps ?: state.config.trancheSteps,
            parentBundleId = lastBundle.bundleId,
            purpose = "continuation ${continuation.continuationId.take(16)} [$SIMULATION_MARKER]",
        )
        state.tranche = next
        state.lifecycle = CandidateLifecycleStatus.RUNNING
        return next
    }

    /**
     * Stop training with explicit semantics (mission §13).
     * SAFE stops at the next atomic boundary and preserves recoverability;
     * EMERGENCY terminates immediately and reports per-component acks.
     */
    fun stopTraining(kind: StopKind): StopResult {
        val state = trainer
        return when (kind) {
            StopKind.SAFE -> {
                val boundary = if (state == null) "no-active-candidate"
                else "accepted-step:${state.globalStep}"
                if (state != null && state.lifecycle == CandidateLifecycleStatus.RUNNING) {
                    state.lifecycle = CandidateLifecycleStatus.PAUSED
                }
                val preserved = mutableListOf<String>()
                state?.latestCheckpoint?.let { preserved += "checkpoint:${it.checkpointId}" }
                state?.pointer?.let { preserved += "pointer:${it.pointerId}" }
                state?.let { preserved += "soul:${it.candidateSoul.soulId}" }
                state?.tranche?.let { preserved += "tranche:${it.trancheId}" }
                auditLog.record(
                    who = "operator", device = "sim", command = "stop_training:safe",
                    target = state?.candidateGenerationId ?: "none", timestamp = now(),
                    previousState = "running", requestedState = "paused",
                    result = "boundary-preserved",
                )
                StopResult.Safe(
                    SafeStopResult(boundary = boundary, preserved = preserved, recoverable = true)
                )
            }
            StopKind.EMERGENCY -> {
                val acks = LinkedHashMap<String, ComponentAck>()
                if (state != null) {
                    state.lifecycle = CandidateLifecycleStatus.PAUSED
                    acks["trainer:${state.candidateGenerationId}"] = ComponentAck.STOPPED
                }
                // Workers that are already lost cannot acknowledge — never
                // claim them stopped.
                for (worker in workers) {
                    acks["worker:${worker.workerId}"] =
                        if (worker.connected) {
                            replaceWorker(worker.copy(connected = false))
                            ComponentAck.STOPPED
                        } else {
                            ComponentAck.UNREACHABLE
                        }
                }
                auditLog.record(
                    who = "operator", device = "sim", command = "stop_training:emergency",
                    target = state?.candidateGenerationId ?: "none", timestamp = now(),
                    previousState = "running", requestedState = "terminated",
                    result = if (EmergencyStopResult(acknowledgements = acks).allStopped) "all-stopped" else "partial",
                )
                StopResult.Emergency(EmergencyStopResult(acknowledgements = acks))
            }
        }
    }

    // ------------------------------------------------------------------
    // Organism-level stop
    // ------------------------------------------------------------------

    /** PAUSE AXON: finish current atomic boundary and prevent new ticks. */
    fun pauseOrganism(): SafeStopResult {
        organismPaused = true
        auditLog.record(
            who = "operator", device = "sim", command = "pause_organism",
            target = "heart", timestamp = now(),
            previousState = "ticking", requestedState = "paused", result = "paused",
        )
        return SafeStopResult(
            boundary = "tick:${field.tickId}",
            preserved = listOf("field:${field.fieldId}") +
                souls.map { (coreId, soul) -> "soul:$coreId:${soul.soulId}" },
        )
    }

    fun resumeOrganism() {
        organismPaused = false
        auditLog.record(
            who = "operator", device = "sim", command = "resume_organism",
            target = "heart", timestamp = now(),
            previousState = "paused", requestedState = "ticking", result = "ticking",
        )
    }

    /** EMERGENCY STOP: attempt to halt everything; report exact acks. */
    fun emergencyStopAll(): EmergencyStopResult {
        organismPaused = true
        val acks = LinkedHashMap<String, ComponentAck>()
        acks["heart"] = ComponentAck.STOPPED
        for (core in activeCores) acks["core:${core.descriptor.coreId}"] = ComponentAck.STOPPED
        trainer?.let {
            it.lifecycle = CandidateLifecycleStatus.PAUSED
            acks["trainer:${it.candidateGenerationId}"] = ComponentAck.STOPPED
        }
        for (worker in workers) {
            acks["worker:${worker.workerId}"] =
                if (worker.connected) {
                    replaceWorker(worker.copy(connected = false))
                    ComponentAck.STOPPED
                } else {
                    ComponentAck.UNREACHABLE
                }
        }
        auditLog.record(
            who = "operator", device = "sim", command = "emergency_stop_all",
            target = "organism", timestamp = now(),
            previousState = "running", requestedState = "terminated",
            result = if (EmergencyStopResult(acknowledgements = acks).allStopped) "all-stopped" else "partial",
        )
        return EmergencyStopResult(acknowledgements = acks)
    }

    // ------------------------------------------------------------------
    // Recovery capsule
    // ------------------------------------------------------------------

    /**
     * Export a self-verifying recovery capsule for the current accepted
     * training boundary. The manifest is always written LAST.
     */
    fun exportRecoveryCapsule(): RecoveryCapsule {
        val state = trainer ?: throw IllegalStateException("no training candidate to export")
        val checkpoint = state.latestCheckpoint
            ?: throw IllegalStateException("no checkpoint available for capsule export")
        val bundle = state.bundles.lastOrNull()
            ?: throw IllegalStateException("no accepted bundle available for capsule export")
        val builder = RecoveryCapsuleBuilder(
            kind = "full-organism-recovery",
            createdAt = now(),
        )
        builder.addFile("anatomy.json", AxonJson.encodeToString(state.anatomy).encodeToByteArray())
        builder.addFile("soul.json", AxonJson.encodeToString(state.candidateSoul).encodeToByteArray())
        builder.addFile("learning_policy.json", AxonJson.encodeToString(state.policy).encodeToByteArray())
        builder.addFile("checkpoint.json", AxonJson.encodeToString(checkpoint).encodeToByteArray())
        builder.addFile("accepted_bundle.json", AxonJson.encodeToString(bundle).encodeToByteArray())
        builder.addFile("tranche.json", AxonJson.encodeToString(state.tranche!!).encodeToByteArray())
        builder.addFile("pointer.json", AxonJson.encodeToString(state.pointer!!).encodeToByteArray())
        builder.addDetail("marker", SIMULATION_MARKER)
        builder.addDetail("candidate_generation_id", state.candidateGenerationId)
        val capsule = builder.build()
        emit(
            AxonEventPayload.RecoveryCapsulePublished(
                capsuleId = capsule.manifest.capsuleId,
                manifestSha256 = capsule.manifest.manifestSha256,
                members = capsule.writeOrder,
            )
        )
        auditLog.record(
            who = "operator", device = "sim", command = "export_recovery_capsule",
            target = state.candidateGenerationId, timestamp = now(),
            previousState = "local", requestedState = "capsule:${capsule.manifest.capsuleId}",
            result = "published",
        )
        return capsule
    }
}
