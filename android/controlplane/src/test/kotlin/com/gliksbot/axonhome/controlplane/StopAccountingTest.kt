package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Stop-ack accounting (arch §13): an emergency result never reports
 * allStopped=true when any component is unreachable — or in any other
 * non-terminal state. The claim must always match the ack map.
 */
class StopAccountingTest {

    private val t = Instant.parse("2026-09-21T00:02:30.500000Z")

    private fun ack(id: String, status: ComponentAckStatus) =
        ComponentAck(id, ComponentKind.COMPUTE_WORKER, status)

    private fun response(components: List<ComponentAck>, claimed: Boolean) = StopResponse(
        freshness = Freshness.LIVE,
        observedAt = t,
        scope = StopScope.EMERGENCY,
        allStopped = claimed,
        components = components,
    )

    @Test
    fun `all confirmed stopped or provably gone is all stopped`() {
        val components = listOf(
            ack("heart", ComponentAckStatus.CONFIRMED_STOPPED),
            ack("trainer", ComponentAckStatus.CONFIRMED_STOPPED),
            ack("gone-worker", ComponentAckStatus.PROVABLY_GONE),
        )
        assertTrue(StopAccounting.computeAllStopped(components))
        assertTrue(StopAccounting.verify(response(components, claimed = true)))
        assertTrue(StopAccounting.unresolved(components).isEmpty())
    }

    @Test
    fun `any unreachable component forces all stopped false`() {
        val components = listOf(
            ack("heart", ComponentAckStatus.CONFIRMED_STOPPED),
            ack("worker-kaggle-01", ComponentAckStatus.UNREACHABLE),
        )
        assertFalse(StopAccounting.computeAllStopped(components))
        assertEquals(listOf("worker-kaggle-01"), StopAccounting.unresolved(components).map { it.componentId })
        // An honest server response verifies...
        assertTrue(StopAccounting.verify(response(components, claimed = false)))
        // ...and a dishonest allStopped=true claim is caught by the client.
        assertFalse(StopAccounting.verify(response(components, claimed = true)))
    }

    @Test
    fun `refused and unavailable also block all stopped`() {
        for (status in listOf(ComponentAckStatus.REFUSED, ComponentAckStatus.UNAVAILABLE)) {
            val components = listOf(
                ack("heart", ComponentAckStatus.CONFIRMED_STOPPED),
                ack("trainer", status),
            )
            assertFalse(StopAccounting.computeAllStopped(components), "$status must block allStopped")
        }
    }

    @Test
    fun `empty component list is never all stopped`() {
        assertFalse(StopAccounting.computeAllStopped(emptyList()))
    }

    @Test
    fun `unreachable dominates even when mixed with provably gone`() {
        val components = listOf(
            ack("old", ComponentAckStatus.PROVABLY_GONE),
            ack("flaky", ComponentAckStatus.UNREACHABLE),
        )
        assertFalse(StopAccounting.computeAllStopped(components))
    }
}
