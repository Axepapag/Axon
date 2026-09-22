package com.gliksbot.axonhome.controlplane

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs

/**
 * Seq-gap detection for the axon-home-event-v1 stream (arch §6.1): a forward
 * jump is reported as a Gap (client re-syncs via REST); replays/out-of-order
 * events are dropped; a visible gap is never silently skipped.
 */
class SeqGapTrackerTest {

    @Test
    fun `in-order stream is contiguous`() {
        val tracker = SeqGapTracker()
        assertIs<SeqGapTracker.Result.InOrder>(tracker.observe(100L))
        (101L..105L).forEach { seq ->
            assertIs<SeqGapTracker.Result.InOrder>(tracker.observe(seq), "seq $seq should be in order")
        }
        assertEquals(105L, tracker.lastSeq)
    }

    @Test
    fun `forward jump is reported as a gap with the exact missed range`() {
        val tracker = SeqGapTracker()
        tracker.observe(10L)
        tracker.observe(11L)
        val result = tracker.observe(15L)
        val gap = assertIs<SeqGapTracker.Result.Gap>(result)
        assertEquals(12L, gap.from)
        assertEquals(14L, gap.to)
        assertEquals(3L, gap.missed) // 12, 13, 14
        assertEquals(15L, tracker.lastSeq)
    }

    @Test
    fun `replayed events are duplicates, not gaps`() {
        val tracker = SeqGapTracker()
        tracker.observe(7L)
        assertIs<SeqGapTracker.Result.Duplicate>(tracker.observe(7L))
        assertIs<SeqGapTracker.Result.Duplicate>(tracker.observe(5L))
        assertIs<SeqGapTracker.Result.InOrder>(tracker.observe(8L))
    }

    @Test
    fun `reset after REST re-sync treats the next event as the first`() {
        val tracker = SeqGapTracker()
        tracker.observe(10L)
        val gap = assertIs<SeqGapTracker.Result.Gap>(tracker.observe(42L))
        assertEquals(31L, gap.missed)
        // Client re-syncs via REST snapshot (arch §6.1), then resumes the stream.
        tracker.reset()
        assertEquals(null, tracker.lastSeq)
        assertIs<SeqGapTracker.Result.InOrder>(tracker.observe(1000L))
    }
}
