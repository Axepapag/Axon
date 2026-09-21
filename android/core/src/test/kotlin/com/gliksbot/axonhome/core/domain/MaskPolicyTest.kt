package com.gliksbot.axonhome.core.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class MaskPolicyTest {

    private fun spansOf(vararg texts: String): List<FieldSpan> =
        texts.mapIndexed { i, t -> FieldSpan(spanId = "s$i", text = t) }

    @Test
    fun `all attends the entire region`() {
        val spans = spansOf("hello", " world")
        assertEquals(listOf(AttendedInterval(0, 11)), resolveMaskPolicy(spans, RegionMaskPolicy.all()))
    }

    @Test
    fun `none attends nothing`() {
        assertEquals(emptyList(), resolveMaskPolicy(spansOf("abc"), RegionMaskPolicy.none()))
    }

    @Test
    fun `last_n_spans attends the newest span intervals`() {
        val spans = spansOf("aa", "bbb", "c", "dd")
        val intervals = resolveMaskPolicy(spans, RegionMaskPolicy.lastNSpans(2))
        // spans "c" (offset 5..6) and "dd" (offset 6..8)
        assertEquals(listOf(AttendedInterval(5, 6), AttendedInterval(6, 8)), intervals)
        assertEquals(emptyList(), resolveMaskPolicy(spans, RegionMaskPolicy.lastNSpans(0)))
    }

    @Test
    fun `tail_percent uses integer ceiling so any nonzero setting exposes at least one char`() {
        val spans = spansOf("0123456789") // 10 chars
        assertEquals(listOf(AttendedInterval(9, 10)), resolveMaskPolicy(spans, RegionMaskPolicy.tailPercent(1)))
        assertEquals(listOf(AttendedInterval(5, 10)), resolveMaskPolicy(spans, RegionMaskPolicy.tailPercent(50)))
        // 101 chars * 10% -> ceil(10.1) = 11
        val long = spansOf("x".repeat(101))
        assertEquals(listOf(AttendedInterval(90, 101)), resolveMaskPolicy(long, RegionMaskPolicy.tailPercent(10)))
    }

    @Test
    fun `tail_percent zero attends nothing and invalid kinds are rejected`() {
        assertEquals(emptyList(), resolveMaskPolicy(spansOf("abc"), RegionMaskPolicy.tailPercent(0)))
        assertFailsWith<IllegalArgumentException> { RegionMaskPolicy.tailPercent(101) }
        assertFailsWith<IllegalArgumentException> { RegionMaskPolicy("blurry") }
    }

    @Test
    fun `mask state defaults attend everything except training regions`() {
        val state = HeartRegionMaskState.defaults()
        assertEquals(RegionMaskPolicy.KIND_ALL, state.policyFor(LogicalRegion.SCRATCH).kind)
        assertEquals(RegionMaskPolicy.KIND_NONE, state.policyFor(LogicalRegion.TRAINER_INSTRUCTIONS).kind)
        assertEquals(RegionMaskPolicy.KIND_NONE, state.policyFor(LogicalRegion.TRAINING_RESPONSES).kind)
    }

    @Test
    fun `mask state revision advances and changes state id`() {
        val before = HeartRegionMaskState.defaults()
        val after = before.withPolicy(LogicalRegion.SCRATCH, RegionMaskPolicy.tailPercent(55))
        assertEquals(before.revision + 1, after.revision)
        kotlin.test.assertNotEquals(before.stateId, after.stateId)
        assertEquals(RegionMaskPolicy.KIND_TAIL_PERCENT, after.policyFor(LogicalRegion.SCRATCH).kind)
    }
}
