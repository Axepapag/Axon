package com.gliksbot.axonhome

import com.gliksbot.axonhome.controlplane.CapsuleExportRequest
import com.gliksbot.axonhome.controlplane.CapsuleKind
import com.gliksbot.axonhome.controlplane.LogicalRegion
import com.gliksbot.axonhome.controlplane.MaskKind
import com.gliksbot.axonhome.controlplane.MaskPolicy
import com.gliksbot.axonhome.controlplane.PutMasksRequest
import com.gliksbot.axonhome.core.capsule.RecoveryCapsuleVerifier
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.sim.ApiException
import com.gliksbot.axonhome.sim.SimControlPlaneClient
import com.gliksbot.axonhome.ui.state.maskChanges
import com.gliksbot.axonhome.ui.state.summarizeCapsuleReport
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class MaskProposalFlowTest {

    private fun rig(): Pair<SimulationEngine, SimControlPlaneClient> {
        val engine = SimulationEngine()
        return engine to SimControlPlaneClient(engine)
    }

    @Test
    fun `first PUT proposes and mutates nothing`() = runTest {
        val (engine, client) = rig()
        val before = engine.maskState
        val response = client.putMasks(
            PutMasksRequest(
                policies = mapOf(LogicalRegion.SCRATCH to MaskPolicy(MaskKind.TAIL_PERCENT, tailPercent = 25)),
                confirmed = false,
            ),
            idempotencyKey = "test-1",
        )
        assertFalse(response.applied)
        assertNotNull(response.confirmToken)
        assertEquals(before.revision, engine.maskState.revision)
        assertEquals(before.stateId, engine.maskState.stateId)
        // The proposal itself reflects the change.
        assertEquals(
            MaskPolicy(MaskKind.TAIL_PERCENT, tailPercent = 25),
            response.proposed.policies[LogicalRegion.SCRATCH],
        )
    }

    @Test
    fun `confirm without token is refused and confirm with token applies`() = runTest {
        val (engine, client) = rig()
        val proposal = client.putMasks(
            PutMasksRequest(
                policies = mapOf(LogicalRegion.DIARY to MaskPolicy(MaskKind.NONE)),
                confirmed = false,
            ),
            idempotencyKey = "test-2",
        )
        try {
            client.putMasks(
                PutMasksRequest(
                    policies = mapOf(LogicalRegion.DIARY to MaskPolicy(MaskKind.NONE)),
                    confirmed = true,
                    confirmToken = "bogus",
                ),
                idempotencyKey = "test-3",
            )
            fail("expected MASK_CONFIRMATION_REQUIRED")
        } catch (e: ApiException) {
            assertEquals(
                com.gliksbot.axonhome.controlplane.ErrorCode.MASK_CONFIRMATION_REQUIRED,
                e.error.code,
            )
        }
        assertEquals(0L, engine.maskState.revision)

        val applied = client.putMasks(
            PutMasksRequest(
                policies = mapOf(LogicalRegion.DIARY to MaskPolicy(MaskKind.NONE)),
                confirmed = true,
                confirmToken = proposal.confirmToken,
            ),
            idempotencyKey = "test-4",
        )
        assertTrue(applied.applied)
        assertEquals(1L, engine.maskState.revision)
    }

    @Test
    fun `maskChanges diffs only real changes`() = runTest {
        val (_, client) = rig()
        val current = client.masks()
        val proposal = client.putMasks(
            PutMasksRequest(
                policies = current.policies + (LogicalRegion.SCRATCH to MaskPolicy(MaskKind.NONE)),
                confirmed = false,
            ),
            idempotencyKey = "test-5",
        )
        val changes = maskChanges(current, proposal.proposed)
        assertEquals(1, changes.size)
        assertTrue(changes[0].contains("scratch"))
        assertTrue(changes[0].contains("none"))
    }
}

class CapsuleVerifyTest {

    private fun trainedRig(): Pair<SimulationEngine, SimControlPlaneClient> {
        val engine = SimulationEngine()
        engine.startTraining()
        repeat(4) { engine.stepTraining() }
        return engine to SimControlPlaneClient(engine)
    }

    @Test
    fun `exported capsule verifies and renders per-member results`() = runTest {
        val (_, client) = trainedRig()
        val manifest = client.exportCapsule(
            CapsuleExportRequest(kind = CapsuleKind.FULL_ORGANISM),
            idempotencyKey = "cap-1",
        )
        assertTrue(manifest.members.isNotEmpty())

        val report = client.verifyCapsule(manifest.capsuleId, idempotencyKey = "cap-2")
        assertTrue(report.ok)
        assertTrue(report.mismatches.isEmpty())

        val summary = summarizeCapsuleReport(report)
        assertTrue(summary.ok)
        assertFalse(summary.quarantined)
        assertTrue(summary.rows.isNotEmpty())
        assertTrue(summary.rows.all { it.ok })
    }

    @Test
    fun `tampered capsule fails closed`() {
        val (engine, _) = trainedRig()
        val capsule = engine.exportRecoveryCapsule()
        val tampered = capsule.files.toMutableMap()
        tampered["soul.json"] = "tampered".encodeToByteArray()
        val verification = RecoveryCapsuleVerifier.verify(tampered, capsule.manifest)
        assertFalse(verification.ok)
        assertTrue(verification.mismatches.isNotEmpty())
    }

    @Test
    fun `export without training refuses honestly`() = runTest {
        val engine = SimulationEngine()
        val client = SimControlPlaneClient(engine)
        try {
            client.exportCapsule(CapsuleExportRequest(kind = CapsuleKind.FULL_ORGANISM), "cap-3")
            fail("expected conflict")
        } catch (e: ApiException) {
            assertEquals(com.gliksbot.axonhome.controlplane.ErrorCode.CONFLICT, e.error.code)
        }
    }
}
