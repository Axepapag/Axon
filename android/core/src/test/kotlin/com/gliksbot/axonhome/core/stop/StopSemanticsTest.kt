package com.gliksbot.axonhome.core.stop

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class StopSemanticsTest {

    @Test
    fun `safe stop reports boundary and preserved recovery state`() {
        val result = SafeStopResult(
            boundary = "accepted-step:7",
            preserved = listOf("checkpoint:abc", "pointer:def", "soul:ghi"),
        )
        assertTrue(result.recoverable)
        assertEquals(3, result.preserved.size)
        assertEquals("accepted-step:7", result.boundary)
    }

    @Test
    fun `emergency stop may only claim all-stopped when every component acked`() {
        val all = EmergencyStopResult(
            acknowledgements = mapOf(
                "heart" to ComponentAck.STOPPED,
                "core:sim-core-1" to ComponentAck.STOPPED,
            )
        )
        assertTrue(all.allStopped)
        assertTrue(all.summary.startsWith("all"))

        val partial = EmergencyStopResult(
            acknowledgements = mapOf(
                "heart" to ComponentAck.STOPPED,
                "worker:kaggle-01" to ComponentAck.UNREACHABLE,
            )
        )
        assertFalse(partial.allStopped)
        assertTrue(partial.summary.contains("worker:kaggle-01=unreachable"))
        assertTrue(partial.summary.startsWith("NOT fully stopped"))
    }

    @Test
    fun `emergency stop with a failed component does not claim all-stopped`() {
        val result = EmergencyStopResult(
            acknowledgements = mapOf(
                "trainer:x" to ComponentAck.FAILED,
            )
        )
        assertFalse(result.allStopped)
    }
}
