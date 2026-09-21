package com.gliksbot.axonhome.core.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class FieldDeltaTest {

    private fun base(): SharedFieldSnapshot = SharedFieldSnapshot.fromTexts(
        mapOf(
            LogicalRegion.SCRATCH to "0123456789",
            LogicalRegion.RESPONSE_DRAFT to "draft",
            LogicalRegion.IDENTITY to "sealed identity",
        )
    )

    private fun delta(snapshot: SharedFieldSnapshot, vararg ops: FieldOperation) = FieldDelta(
        baseFieldId = snapshot.fieldId,
        baseTickId = snapshot.tickId,
        authorCoreId = "sim-core-1",
        passId = "consolidated",
        operations = ops.toList(),
    )

    @Test
    fun `stale delta is rejected`() {
        val snapshot = base()
        val stale = delta(snapshot, InsertText(LogicalRegion.SCRATCH, 0, "x"))
            .copy(baseFieldId = "not-the-current-field")
        assertFailsWith<StaleDeltaError> { validateDelta(snapshot, stale) }
        val wrongTick = delta(snapshot, InsertText(LogicalRegion.SCRATCH, 0, "x"))
            .copy(baseTickId = snapshot.tickId + 5)
        assertFailsWith<StaleDeltaError> { validateDelta(snapshot, wrongTick) }
    }

    @Test
    fun `sealed region write is rejected`() {
        val snapshot = base()
        val write = delta(snapshot, InsertText(LogicalRegion.IDENTITY, 0, "x"))
        assertFailsWith<SealedRegionWriteError> { validateDelta(snapshot, write) }
    }

    @Test
    fun `out of bounds operation is rejected`() {
        val snapshot = base()
        val oob = delta(snapshot, DeleteText(LogicalRegion.SCRATCH, 8, 99))
        assertFailsWith<DeltaValidationError> { validateDelta(snapshot, oob) }
    }

    @Test
    fun `overlapping operations are rejected`() {
        val snapshot = base()
        val overlapping = delta(
            snapshot,
            ReplaceText(LogicalRegion.SCRATCH, 2, 6, "AA"),
            DeleteText(LogicalRegion.SCRATCH, 4, 8),
        )
        assertFailsWith<OverlappingDeltaError> { validateDelta(snapshot, overlapping) }
        // Two inserts at the same offset also overlap.
        val sameOffsetInserts = delta(
            snapshot,
            InsertText(LogicalRegion.SCRATCH, 3, "a"),
            InsertText(LogicalRegion.SCRATCH, 3, "b"),
        )
        assertFailsWith<OverlappingDeltaError> { validateDelta(snapshot, sameOffsetInserts) }
    }

    @Test
    fun `apply insert replace and delete deterministically`() {
        val snapshot = base()
        val d = delta(
            snapshot,
            InsertText(LogicalRegion.SCRATCH, 0, ">>"),
            ReplaceText(LogicalRegion.SCRATCH, 8, 10, "END"),
            DeleteText(LogicalRegion.RESPONSE_DRAFT, 0, 5),
        )
        val next = applyDelta(snapshot, d)
        assertEquals(">>01234567END", next.region(LogicalRegion.SCRATCH).text)
        assertEquals("", next.region(LogicalRegion.RESPONSE_DRAFT).text)
        assertEquals("sealed identity", next.region(LogicalRegion.IDENTITY).text)
        // Re-applying the same delta to the successor is stale.
        assertFailsWith<StaleDeltaError> { validateDelta(next, d) }
    }

    @Test
    fun `delta applied spans carry delta provenance`() {
        val snapshot = base()
        val d = delta(snapshot, InsertText(LogicalRegion.SCRATCH, 0, "note"))
        val next = applyDelta(snapshot, d)
        val inserted = next.region(LogicalRegion.SCRATCH).spans.first { it.text == "note" }
        assertEquals("delta_insert", inserted.kind)
        assertEquals("sim-core-1", inserted.source)
        assertEquals("field_delta:${d.deltaId}:op:0", inserted.provenance)
        assertEquals(64, d.deltaId.length)
    }

    @Test
    fun `permitted regions widen authority for heart grants`() {
        val snapshot = base()
        val write = delta(snapshot, DeleteText(LogicalRegion.IDENTITY, 0, 6))
        assertFailsWith<SealedRegionWriteError> { validateDelta(snapshot, write) }
        val next = applyDelta(snapshot, write, permittedRegions = setOf(LogicalRegion.IDENTITY))
        assertEquals(" identity", next.region(LogicalRegion.IDENTITY).text)
    }

    @Test
    fun `replay applies deltas in order`() {
        var snapshot = base()
        val d1 = delta(snapshot, InsertText(LogicalRegion.SCRATCH, 10, "!"))
        val s1 = applyDelta(snapshot, d1)
        val d2 = FieldDelta(
            baseFieldId = s1.fieldId, baseTickId = s1.tickId,
            authorCoreId = "sim-core-1", passId = "consolidated",
            operations = listOf(InsertText(LogicalRegion.SCRATCH, 11, "?")),
        )
        val final = replayDeltas(snapshot, listOf(d1, d2))
        assertEquals("0123456789!?", final.region(LogicalRegion.SCRATCH).text)
        assertEquals(snapshot.tickId + 2, final.tickId)
    }
}
