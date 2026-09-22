package com.gliksbot.axonhome.core.sim

import com.gliksbot.axonhome.core.domain.CandidateLifecycleStatus
import com.gliksbot.axonhome.core.domain.LogicalRegion
import com.gliksbot.axonhome.core.domain.ROLLING_RETENTION
import com.gliksbot.axonhome.core.domain.SoulTemperature
import com.gliksbot.axonhome.core.domain.StateFreshness
import com.gliksbot.axonhome.core.events.AxonEventPayload
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SimulationEngineTest {

    @Test
    fun `tick emits events in the exact pipeline order`() {
        val engine = SimulationEngine(seed = 7)
        engine.submitUserInput("hello axon")
        engine.eventLog.events.size // baseline (ingress does not emit tick events)
        val record = engine.tick()
        assertNotNull(record)

        val types = engine.eventLog.events
            .filter {
                val p = it.payload
                (p is AxonEventPayload.TickStarted && p.tickId == record.tickNumber) ||
                    (p is AxonEventPayload.CoreFirstReturned && p.tickId == record.tickNumber) ||
                    (p is AxonEventPayload.FirstBarrierComplete && p.tickId == record.tickNumber) ||
                    (p is AxonEventPayload.CoreRefinedReturned && p.tickId == record.tickNumber) ||
                    (p is AxonEventPayload.RefinedBarrierComplete && p.tickId == record.tickNumber) ||
                    (p is AxonEventPayload.ConsolidatorSelected && p.tickId == record.tickNumber) ||
                    (p is AxonEventPayload.FinalReturned && p.tickId == record.tickNumber) ||
                    (p is AxonEventPayload.HeartValidationStarted && p.tickId == record.tickNumber) ||
                    (p is AxonEventPayload.FieldCommitted && p.tickId == record.tickNumber)
            }.map { it.type }

        assertEquals("TickStarted", types.first())
        // FIRST returns (4 cores) then the FIRST barrier.
        assertEquals(
            List(4) { "CoreFirstReturned" },
            types.subList(1, 5),
        )
        assertEquals("FirstBarrierComplete", types[5])
        assertEquals(List(4) { "CoreRefinedReturned" }, types.subList(6, 10))
        assertEquals("RefinedBarrierComplete", types[10])
        assertEquals("ConsolidatorSelected", types[11])
        assertEquals("FinalReturned", types[12])
        assertEquals("HeartValidationStarted", types[13])
        if (!record.rejected) {
            assertEquals("FieldCommitted", types.last())
            // FieldCommitted is the last event of the tick.
            val committedIdx = engine.eventLog.events.indexOfLast {
                it.payload is AxonEventPayload.FieldCommitted &&
                    (it.payload as AxonEventPayload.FieldCommitted).tickId == record.tickNumber
            }
            val tickStartIdx = engine.eventLog.events.indexOfFirst {
                it.payload is AxonEventPayload.TickStarted &&
                    (it.payload as AxonEventPayload.TickStarted).tickId == record.tickNumber
            }
            assertTrue(committedIdx > tickStartIdx)
            val laterTickEvents = engine.eventLog.events.drop(committedIdx + 1)
                .filter { it.type == "TickStarted" }
            // no subsequent TickStarted in this single-tick run
            assertTrue(laterTickEvents.isEmpty())
        }
    }

    @Test
    fun `same seed produces identical field id chain`() {
        fun run(seed: Long): List<String> {
            val engine = SimulationEngine(seed = seed)
            engine.submitUserInput("status report please")
            repeat(6) { engine.tick() }
            return engine.tickHistory.mapNotNull { it.resultFieldId } + engine.field.fieldId
        }
        assertEquals(run(42L), run(42L))
        assertNotEquals(run(42L), run(43L))
    }

    @Test
    fun `consolidator rotates with (tick-1) mod n over eligible cores`() {
        val engine = SimulationEngine(seed = 11)
        // Force acceptance regardless of seeded rejections by replaying until
        // we have 6 committed ticks; rotation is derived from tick numbers.
        val seen = mutableListOf<String>()
        var guard = 0
        while (seen.size < 6 && guard++ < 60) {
            val record = engine.tick() ?: continue
            if (!record.rejected) seen += record.consolidatorCoreId
        }
        assertEquals(6, seen.size)
        val pool = engine.cores.filter { it.consolidatorEligible }.map { it.descriptor.coreId }
        assertEquals(3, pool.size) // 4 heterogeneous cores, one consolidator-ineligible
        val tickNumbers = engine.tickHistory.filter { !it.rejected }.map { it.tickNumber }
        seen.forEachIndexed { i, coreId ->
            assertEquals(pool[(tickNumbers[i] - 1) % pool.size], coreId)
        }
        // Rotation actually visits different cores.
        assertTrue(seen.toSet().size > 1)
        // The ineligible core never consolidates.
        assertFalse(seen.contains(engine.cores.last().descriptor.coreId))
    }

    @Test
    fun `user input is materialized into conversation history and cleared`() {
        val engine = SimulationEngine(seed = 5)
        engine.submitUserInput("what is your status")
        assertTrue(engine.field.region(LogicalRegion.USER_INPUT).text.contains("what is your status"))
        var record = engine.tick()
        var guard = 0
        while (record != null && record.rejected && guard++ < 20) record = engine.tick()
        assertNotNull(record)
        assertFalse(record.rejected)
        assertEquals("", engine.field.region(LogicalRegion.USER_INPUT).text.trim())
        assertTrue(engine.field.region(LogicalRegion.CONVERSATION_HISTORY).text.contains("what is your status"))
    }

    @Test
    fun `soul generations advance two passes per tick plus consolidator finalization`() {
        val engine = SimulationEngine(seed = 9)
        // Rejected ticks (seeded failures) still burn first/refined passes, so
        // measure deltas around the single successful tick.
        var before: Map<String, Int>
        var record: TickRecord
        var guard = 0
        while (true) {
            check(guard++ < 40) { "no successful tick" }
            before = engine.cores.associate { it.descriptor.coreId to engine.soulOf(it.descriptor.coreId).generation }
            record = engine.tick()!!
            if (!record.rejected) break
        }
        for (core in engine.cores) {
            val expected = before.getValue(core.descriptor.coreId) +
                if (core.descriptor.coreId == record.consolidatorCoreId) 3 else 2
            assertEquals(expected, engine.soulOf(core.descriptor.coreId).generation)
        }
        // Receipt chain is linear and the consolidator's last receipt is bound
        // to the canonical successor field.
        val receipts = engine.soulReceiptsOf(record.consolidatorCoreId)
        assertTrue(receipts.size >= 3)
        assertEquals(receipts.last().commitBinding, "canonical-field:${engine.field.fieldId}")
    }

    @Test
    fun `trainer keeps rolling-3 checkpoints and pointer window`() {
        val engine = SimulationEngine(seed = 13)
        engine.startTraining(SimTrainingConfig(trancheSteps = 10))
        repeat(7) { assertTrue(engine.stepTraining()) }
        val trainer = engine.trainer!!
        assertEquals(7, trainer.globalStep)
        assertEquals(ROLLING_RETENTION, trainer.retainedCheckpoints.size)
        // Retained checkpoints are the newest ones.
        assertEquals(listOf(5, 6, 7), trainer.retainedCheckpoints.map { it.step })
        // Lineage chain intact inside the window.
        assertEquals(
            trainer.retainedCheckpoints[1].checkpointId,
            trainer.retainedCheckpoints[2].previousCheckpointId,
        )
        assertEquals(ROLLING_RETENTION, trainer.pointer!!.rollingBundleIds.size)
        assertEquals(7, trainer.pointer!!.currentStep)
    }

    @Test
    fun `tranche exhaustion pauses and continuation resumes`() {
        val engine = SimulationEngine(seed = 17)
        engine.startTraining(SimTrainingConfig(trancheSteps = 3))
        repeat(3) { assertTrue(engine.stepTraining()) }
        assertFalse(engine.stepTraining()) // tranche exhausted
        assertEquals(CandidateLifecycleStatus.PAUSED, engine.trainer!!.lifecycle)
        assertTrue(
            engine.eventLog.ofType("AlertRaised").any {
                (it.payload as AxonEventPayload.AlertRaised).message.contains("tranche")
            }
        )
        engine.requestNextTranche(steps = 2)
        assertEquals(CandidateLifecycleStatus.RUNNING, engine.trainer!!.lifecycle)
        assertTrue(engine.stepTraining())
        assertEquals(4, engine.trainer!!.globalStep)
        assertEquals(1, engine.trainer!!.continuations.size)
    }

    @Test
    fun `heldout evaluation completes at the configured interval`() {
        val engine = SimulationEngine(seed = 19)
        engine.startTraining(SimTrainingConfig(trancheSteps = 12, evalInterval = 4))
        repeat(8) { engine.stepTraining() }
        assertEquals(2, engine.trainer!!.heldoutEvaluations.size)
        assertEquals(
            2,
            engine.eventLog.ofType("HeldoutEvaluationCompleted").size,
        )
    }

    @Test
    fun `capsule export publishes event and verifies cleanly`() {
        val engine = SimulationEngine(seed = 23)
        engine.startTraining(SimTrainingConfig(trancheSteps = 4))
        repeat(3) { engine.stepTraining() }
        val capsule = engine.exportRecoveryCapsule()
        // Manifest is written last.
        assertEquals("manifest.json", capsule.writeOrder.last())
        assertTrue(engine.eventLog.ofType("RecoveryCapsulePublished").isNotEmpty())
        val verification = com.gliksbot.axonhome.core.capsule.RecoveryCapsuleVerifier.verify(
            capsule.files, capsule.manifest,
            expectedArchitectureId = engine.trainer!!.anatomy.architectureId,
            expectedCandidateGenerationId = engine.trainer!!.candidateGenerationId,
        )
        assertTrue(verification.ok, verification.mismatches.toString())
        assertTrue(verification.checksPassed.contains("lineage"))
    }

    @Test
    fun `capsule verification fails closed on a tampered member`() {
        val engine = SimulationEngine(seed = 29)
        engine.startTraining(SimTrainingConfig(trancheSteps = 4))
        repeat(2) { engine.stepTraining() }
        val capsule = engine.exportRecoveryCapsule()
        val tampered = capsule.files.toMutableMap()
        tampered["checkpoint.json"] = tampered.getValue("checkpoint.json") + " ".encodeToByteArray()
        val verification = com.gliksbot.axonhome.core.capsule.RecoveryCapsuleVerifier.verify(
            tampered, capsule.manifest,
        )
        assertFalse(verification.ok)
        assertTrue(verification.mismatches.any { it.contains("checkpoint.json") })
    }

    @Test
    fun `capsule verification fails closed on a forged member-set`() {
        val engine = SimulationEngine(seed = 31)
        engine.startTraining(SimTrainingConfig(trancheSteps = 2))
        repeat(2) { engine.stepTraining() }
        val capsule = engine.exportRecoveryCapsule()
        val forged = capsule.files + ("evil.json" to "junk".encodeToByteArray())
        val verification = com.gliksbot.axonhome.core.capsule.RecoveryCapsuleVerifier.verify(
            forged, capsule.manifest,
        )
        assertFalse(verification.ok)
        assertTrue(verification.mismatches.any { it.contains("evil.json") })
    }

    @Test
    fun `mirrors are freshness labelled and simulation marked`() {
        val engine = SimulationEngine(seed = 37)
        val mirror = engine.fieldMirror(StateFreshness.CACHED)
        assertEquals(StateFreshness.CACHED, mirror.freshness)
        assertEquals(SIMULATION_MARKER, mirror.source)
        // Canonical state carries the SIMULATION banner.
        assertTrue(engine.field.region(LogicalRegion.IDENTITY).text.contains(SIMULATION_MARKER))
        // Event ids are deterministic and sequential.
        assertEquals(engine.eventLog.events.indices.toList(), engine.eventLog.events.map { it.seq.toInt() })
    }

    @Test
    fun `pause organism suppresses ticks and resume restarts them`() {
        val engine = SimulationEngine(seed = 41)
        engine.pauseOrganism()
        assertNull(engine.tick())
        engine.resumeOrganism()
        assertNotNull(engine.tick())
    }

    @Test
    fun `training cannot start twice concurrently`() {
        val engine = SimulationEngine(seed = 43)
        engine.startTraining()
        assertFailsWith<IllegalArgumentException> { engine.startTraining() }
    }

    @Test
    fun `audit log records operator mutations`() {
        val engine = SimulationEngine(seed = 47)
        engine.submitUserInput("hello")
        engine.startTraining()
        val commands = engine.auditLog.entries.map { it.command }
        assertTrue("submit_user_input" in commands)
        assertTrue("start_training" in commands)
        assertTrue(engine.auditLog.entries.all { it.who == "operator" })
    }
}
