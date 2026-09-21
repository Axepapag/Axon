package com.gliksbot.axonhome.core.events

import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString
import com.gliksbot.axonhome.core.domain.AxonJson
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class EventLogTest {

    @Test
    fun `sequence assignment is monotonic from zero`() {
        val log = EventLog()
        log.append(AxonEventPayload.TickStarted(1, "f0"), "2026-02-16T00:00:00Z")
        log.append(AxonEventPayload.AlertRaised("info", "m", "SIMULATION"), "2026-02-16T00:00:01Z")
        assertEquals(listOf(0L, 1L), log.events.map { it.seq })
        assertEquals(2L, log.nextSeq)
    }

    @Test
    fun `replay from seq returns the suffix in order`() {
        val log = EventLog()
        repeat(5) { log.append(AxonEventPayload.TickStarted(it, "f$it"), "2026-02-16T00:00:0${it}Z") }
        assertEquals(listOf(3L, 4L), log.replayFrom(3).map { it.seq })
        assertEquals(5, log.replayFrom(0).size)
        assertTrue(log.replayFrom(99).isEmpty())
    }

    @Test
    fun `event envelope carries schema type and typed payload`() {
        val log = EventLog()
        val event = log.append(AxonEventPayload.ConsolidatorSelected(3, "sim-core-2"), "2026-02-16T00:00:00Z")
        assertEquals("axon-home-event-v1", event.schema)
        assertEquals("ConsolidatorSelected", event.type)
        val text = AxonJson.encodeToString(event)
        assertTrue(text.contains("\"type\":\"ConsolidatorSelected\""))
        val decoded = AxonJson.decodeFromString<AxonEvent>(text)
        assertEquals(event, decoded)
        assertIs<AxonEventPayload.ConsolidatorSelected>(decoded.payload)
    }

    @Test
    fun `all eighteen event payload types roundtrip through json`() {
        val payloads: List<AxonEventPayload> = listOf(
            AxonEventPayload.TickStarted(1, "f"),
            AxonEventPayload.CoreFirstReturned(1, "c", "p"),
            AxonEventPayload.FirstBarrierComplete(1, listOf("c")),
            AxonEventPayload.CoreRefinedReturned(1, "c", "p"),
            AxonEventPayload.RefinedBarrierComplete(1, listOf("c")),
            AxonEventPayload.ConsolidatorSelected(1, "c"),
            AxonEventPayload.FinalReturned(1, "c", "v"),
            AxonEventPayload.HeartValidationStarted(1, "v"),
            AxonEventPayload.FieldCommitted(1, "f1", "f0", listOf("d")),
            AxonEventPayload.SoulTransitionAccepted("c", "s", 1, "r", "first"),
            AxonEventPayload.TrainerStepAccepted("m", "cand", 1, 1.5, 0.4, "b"),
            AxonEventPayload.CheckpointWritten("cp", 1, "cand"),
            AxonEventPayload.HeldoutEvaluationCompleted("e", "cand", mapOf("text_exact_rate" to 1.0)),
            AxonEventPayload.ComputeWorkerConnected("w", "kaggle"),
            AxonEventPayload.ComputeWorkerLost("w", "timeout"),
            AxonEventPayload.RecoveryCapsulePublished("cap", "sha", listOf("m")),
            AxonEventPayload.AgentMessageReceived("codex", "roundtable", "hi"),
            AxonEventPayload.AlertRaised("warning", "m", "SIMULATION"),
        )
        assertEquals(18, payloads.size)
        for (payload in payloads) {
            val text = AxonJson.encodeToString(payload)
            val decoded = AxonJson.decodeFromString<AxonEventPayload>(text)
            assertEquals(payload, decoded)
            assertTrue(text.contains("\"type\":\"${payload.eventType}\""), "missing type tag in $text")
        }
    }
}
