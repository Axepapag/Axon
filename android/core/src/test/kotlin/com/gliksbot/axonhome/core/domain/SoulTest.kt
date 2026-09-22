package com.gliksbot.axonhome.core.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class SoulTest {

    private fun soul(): SoulSnapshot = genesisSoul("sim-core-1", "living-d64-english-x", "untrained", "s")

    @Test
    fun `genesis soul has four ordered temperature layers`() {
        val s = soul()
        assertEquals(listOf(SoulTemperature.HOT, SoulTemperature.WARM, SoulTemperature.COLD, SoulTemperature.DEEP_COLD),
            s.layers.map { it.temperature })
        assertEquals(0, s.generation)
        assertEquals(null, s.parentSoulId)
        assertEquals("axon-private-soul-snapshot-v1", s.schema)
        assertEquals("axon-private-soul-layer-v1", s.layers.first().schema)
    }

    @Test
    fun `transition increments generation and chains parent`() {
        val before = soul()
        val transition = SoulTransition(
            coreId = before.coreId, branchId = "live", beforeSoulId = before.soulId,
            phase = "first", updatedTemperatures = listOf(SoulTemperature.HOT),
        )
        val (after, receipt) = applySoulTransition(before, transition)
        assertEquals(before.generation + 1, after.generation)
        assertEquals(before.soulId, after.parentSoulId)
        assertEquals(receipt.beforeSoulId, before.soulId)
        assertEquals(receipt.afterSoulId, after.soulId)
        assertEquals(after.generation, receipt.generation)
        assertEquals(transition.transitionId, receipt.transitionId)
        // HOT payload changed; colder layers untouched.
        assertNotEquals(before.layer(SoulTemperature.HOT).payloadSha256, after.layer(SoulTemperature.HOT).payloadSha256)
        assertEquals(before.layer(SoulTemperature.COLD).payloadSha256, after.layer(SoulTemperature.COLD).payloadSha256)
    }

    @Test
    fun `stale transition is rejected (before idempotence)`() {
        val before = soul()
        val transition = SoulTransition(before.coreId, "live", before.soulId, "first", listOf(SoulTemperature.HOT))
        val (after, _) = applySoulTransition(before, transition)
        assertFailsWith<SoulTransitionError> { applySoulTransition(after, transition) }
    }

    @Test
    fun `runtime phases must update the HOT layer`() {
        val before = soul()
        assertFailsWith<IllegalArgumentException> {
            SoulTransition(before.coreId, "live", before.soulId, "refined", listOf(SoulTemperature.WARM))
        }
    }

    @Test
    fun `receipt chain is linear across repeated transitions`() {
        var current = soul()
        val receipts = mutableListOf<SoulCommitReceipt>()
        repeat(5) {
            val transition = SoulTransition(current.coreId, "live", current.soulId, "first", listOf(SoulTemperature.HOT))
            val (next, receipt) = applySoulTransition(current, transition)
            receipts += receipt
            current = next
        }
        assertEquals(5, current.generation)
        for (i in 1 until receipts.size) {
            assertEquals(receipts[i - 1].afterSoulId, receipts[i].beforeSoulId)
        }
    }

    @Test
    fun `anatomy is d64 locked and architecture id has the canonical prefix`() {
        val anatomy = AnatomyConfig(nHeads = 1, nLayers = 2, ffnDim = 131072, stateTokens = 4, pageSize = 32)
        assertTrue(anatomy.architectureId.startsWith("living-d64-english-"))
        assertEquals("living-d64-english-".length + 24, anatomy.architectureId.length)
        assertFailsWith<IllegalArgumentException> {
            AnatomyConfig(dModel = 32, nHeads = 1, nLayers = 1, ffnDim = 16, stateTokens = 4, pageSize = 32)
        }
        // Same config → same architecture id.
        assertEquals(anatomy.architectureId,
            AnatomyConfig(nHeads = 1, nLayers = 2, ffnDim = 131072, stateTokens = 4, pageSize = 32).architectureId)
    }
}
