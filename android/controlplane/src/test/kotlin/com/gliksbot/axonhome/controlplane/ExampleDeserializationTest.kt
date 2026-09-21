package com.gliksbot.axonhome.controlplane

import kotlinx.serialization.json.Json
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Contract test: every example in the repo-level controlplane/examples/ tree
 * deserializes into the Kotlin client models, and the wire contract details
 * the architecture doc demands are actually visible in the models.
 *
 * This is the primary examples<->models consistency enforcement (see also
 * SchemaResourcesConsistencyTest for schemas<->resources consistency).
 */
class ExampleDeserializationTest {

    private val json = Json { ignoreUnknownKeys = true }

    private fun load(name: String): String {
        val file = ControlPlanePaths.example(name)
        assertTrue(file.isFile, "example file missing: ${file.absolutePath}")
        return file.readText()
    }

    @Test
    fun `health example deserializes`() {
        val health = json.decodeFromString<HealthResponse>(load("health.json"))
        assertEquals(HealthResponse.SCHEMA, health.schema)
        assertEquals(Freshness.LIVE, health.freshness)
        assertEquals("ok", health.controlPlane.status)
        val heart = assertNotNull(health.heart)
        assertEquals("axon-heart-health-v2", heart.schema)
        assertEquals(1188, heart.tickSequence)
        assertTrue(heart.lastBeatOk)
    }

    @Test
    fun `trainer command UNAVAILABLE passthrough deserializes`() {
        val response = json.decodeFromString<TrainerCommandResponse>(load("trainer_command_unavailable.json"))
        assertEquals(TrainerCommandKind.PAUSE, response.kind)
        // Fail-closed passthrough (arch §5.5): UNAVAILABLE is preserved verbatim,
        // never rewritten as success.
        assertEquals(TrainerCommandStatus.UNAVAILABLE, response.status)
        assertEquals("no governed handler; no action was taken", response.message)
        assertNull(response.result)
        assertFalse(response.idempotentReplay)
    }

    @Test
    fun `TickStarted event envelope decodes into sealed hierarchy`() {
        val event = AxonEvent.json.decodeFromString<AxonEvent>(load("event_tick_started.json"))
        val tickStarted = assertIs<AxonEvent.TickStarted>(event)
        assertEquals(AxonEvent.ENVELOPE_SCHEMA, tickStarted.schema)
        assertEquals(10234, tickStarted.seq)
        assertEquals("axon-heart-tick-identity-v1", tickStarted.payload.schema)
        assertEquals(1189, tickStarted.payload.tickSequence)
        assertEquals(4, tickStarted.payload.participants.size)
    }

    @Test
    fun `capsule manifest example deserializes with repo schema string`() {
        val manifest = json.decodeFromString<CapsuleManifest>(load("capsule_manifest.json"))
        assertEquals("axon-cloud-bundle-manifest-v1", manifest.schema)
        assertEquals(CapsuleKind.FULL_ORGANISM, manifest.kind)
        assertEquals(8, manifest.members.size)
        assertTrue(manifest.members.values.all { it.sha256.isNotBlank() && it.bytes > 0 })
        assertEquals("843bbedd25347d367da1f979488a6ecb77128ab7", manifest.gitCommit)
    }

    @Test
    fun `emergency stop example with unreachable component is not all stopped`() {
        val stop = json.decodeFromString<StopResponse>(load("stop_emergency_partial.json"))
        assertEquals(StopScope.EMERGENCY, stop.scope)
        assertEquals(4, stop.components.size)
        // Honest ack accounting (arch §13): one unreachable component means
        // all_stopped MUST be false, and the claim must match the ack map.
        assertFalse(stop.allStopped)
        assertTrue(StopAccounting.verify(stop))
        val unreachable = StopAccounting.unresolved(stop.components)
        assertEquals(2, unreachable.size) // trainer UNAVAILABLE + worker UNREACHABLE
        assertTrue(unreachable.any { it.status == ComponentAckStatus.UNREACHABLE })
    }

    @Test
    fun `audit entry example deserializes`() {
        val entry = json.decodeFromString<AuditEntry>(load("audit_entry.json"))
        assertEquals(AuditEntry.SCHEMA, entry.schema)
        assertEquals("jeff", entry.who)
        assertEquals(AuditResult.UNAVAILABLE, entry.result)
        assertEquals("no governed handler; no action was taken", entry.error)
    }

    @Test
    fun `mask proposal example is a two-step confirm, not an applied mutation`() {
        val response = json.decodeFromString<PutMasksResponse>(load("mask_proposal.json"))
        // arch §5.5/§10.F: first touch never mutates.
        assertFalse(response.applied)
        assertNotNull(response.confirmToken)
        // Identity region is always fully attended (IDENTITY_MASK_POLICY).
        assertEquals(MaskKind.ALL, response.proposed.policies[LogicalRegion.IDENTITY]?.kind)
        assertEquals(13, response.proposed.policies.size)
        assertEquals(
            MaskPolicy(MaskKind.TAIL_PERCENT, tailPercent = 50),
            response.proposed.policies[LogicalRegion.CONVERSATION_HISTORY],
        )
    }

    @Test
    fun `all example files are covered by this test`() {
        val covered = setOf(
            "health.json",
            "trainer_command_unavailable.json",
            "event_tick_started.json",
            "capsule_manifest.json",
            "stop_emergency_partial.json",
            "audit_entry.json",
            "mask_proposal.json",
        )
        val onDisk = ControlPlanePaths.examplesDir()
            .listFiles { f -> f.extension == "json" }!!
            .map { it.name }
            .toSet()
        val uncovered = onDisk - covered
        assertTrue(uncovered.isEmpty(), "example files without a deserialization test: $uncovered")
    }

    @Test
    fun `examples dir resolution fails loudly with a useful message`() {
        // Sanity: the resolver found a real tree (otherwise every test above failed),
        // and an unrelated directory is rejected.
        assertFailsWith<IllegalStateException> {
            val bogus = java.io.File("/definitely/not/the/repo")
            check(File(bogus, "examples").isDirectory) { "Cannot locate the repo-level controlplane/ tree" }
        }
    }
}
