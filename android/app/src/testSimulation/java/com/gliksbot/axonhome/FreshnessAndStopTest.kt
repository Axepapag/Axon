package com.gliksbot.axonhome

import com.gliksbot.axonhome.controlplane.ComponentAck
import com.gliksbot.axonhome.controlplane.ComponentAckStatus
import com.gliksbot.axonhome.controlplane.ComponentKind
import com.gliksbot.axonhome.controlplane.Freshness
import com.gliksbot.axonhome.controlplane.StopResponse
import com.gliksbot.axonhome.controlplane.StopScope
import com.gliksbot.axonhome.ui.state.freshnessLabel
import com.gliksbot.axonhome.ui.state.summarizeStop
import kotlinx.datetime.Instant
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FreshnessMappingTest {

    @Test
    fun `freshness labels render sim prefix`() {
        assertEquals("SIMULATION · LIVE", freshnessLabel(Freshness.LIVE, simulation = true))
        assertEquals("CACHED", freshnessLabel(Freshness.CACHED, simulation = false))
        assertEquals("SIMULATION · STALE", freshnessLabel(Freshness.STALE, simulation = true))
        assertEquals("DISCONNECTED", freshnessLabel(Freshness.DISCONNECTED, simulation = false))
    }
}

class StopAckRenderingTest {

    private fun response(allStopped: Boolean, statuses: List<ComponentAckStatus>): StopResponse =
        StopResponse(
            freshness = Freshness.LIVE,
            observedAt = Instant.fromEpochSeconds(1_770_000_000),
            scope = StopScope.EMERGENCY,
            allStopped = allStopped,
            components = statuses.mapIndexed { index, status ->
                ComponentAck(
                    componentId = "component-$index",
                    componentKind = if (index == 0) ComponentKind.HEART_HOST else ComponentKind.COMPUTE_WORKER,
                    status = status,
                    detail = if (status == ComponentAckStatus.UNREACHABLE) "no route" else null,
                )
            },
        )

    @Test
    fun `unreachable component blocks the all-stopped headline`() {
        val summary = summarizeStop(
            response(
                allStopped = false,
                statuses = listOf(
                    ComponentAckStatus.CONFIRMED_STOPPED,
                    ComponentAckStatus.UNREACHABLE,
                ),
            ),
        )
        assertFalse(summary.allStopped)
        assertTrue(summary.claimVerified)
        assertEquals(2, summary.rows.size)
        assertTrue(summary.rows[1].blocksAllStopped)
        assertEquals("unreachable", summary.rows[1].statusLabel)
        assertTrue(summary.headline.startsWith("NOT everything is stopped"))
    }

    @Test
    fun `a lying all-stopped claim is detected and overridden`() {
        val summary = summarizeStop(
            response(
                allStopped = true, // server lies
                statuses = listOf(
                    ComponentAckStatus.CONFIRMED_STOPPED,
                    ComponentAckStatus.UNREACHABLE,
                ),
            ),
        )
        assertFalse(summary.claimVerified)
        assertFalse(summary.allStopped)
        assertTrue(summary.headline.startsWith("WARNING"))
    }

    @Test
    fun `all confirmed stopped renders all-stopped`() {
        val summary = summarizeStop(
            response(
                allStopped = true,
                statuses = listOf(
                    ComponentAckStatus.CONFIRMED_STOPPED,
                    ComponentAckStatus.PROVABLY_GONE,
                ),
            ),
        )
        assertTrue(summary.allStopped)
        assertTrue(summary.claimVerified)
        assertTrue(summary.rows.none { it.blocksAllStopped })
    }
}
