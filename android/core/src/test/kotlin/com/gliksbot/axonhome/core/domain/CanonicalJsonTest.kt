package com.gliksbot.axonhome.core.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CanonicalJsonTest {

    @Test
    fun `canonical json sorts keys and omits whitespace`() {
        val value = mapOf("b" to 1, "a" to listOf("x", true, null))
        assertEquals("""{"a":["x",true,null],"b":1}""", CanonicalJson.encode(value))
    }

    @Test
    fun `canonical json escapes control characters like python`() {
        assertEquals("\"a\\nb\\u0001c\"", CanonicalJson.encode("a\nb\u0001c"))
    }

    @Test
    fun `canonical json renders integral doubles with trailing point zero`() {
        assertEquals("1.0", CanonicalJson.encode(1.0))
        assertEquals("0.5", CanonicalJson.encode(0.5))
    }

    @Test
    fun `canonical json rejects non finite doubles`() {
        assertFailsWith<IllegalArgumentException> { CanonicalJson.encode(Double.NaN) }
        assertFailsWith<IllegalArgumentException> { CanonicalJson.encode(Double.POSITIVE_INFINITY) }
    }

    @Test
    fun `canonical sha256 is deterministic and order independent`() {
        val a = canonicalSha256(mapOf("x" to 1, "y" to "z"))
        val b = canonicalSha256(mapOf("y" to "z", "x" to 1))
        assertEquals(a, b)
        assertEquals(64, a.length)
    }
}

class SharedFieldModelTest {

    @Test
    fun `thirteen canonical regions in canonical order`() {
        assertEquals(13, CANONICAL_REGION_ORDER.size)
        assertEquals(
            listOf(
                "conversation_history", "user_input", "cortex", "situation_awareness",
                "tool_results", "advisor_input", "task_state", "scratch",
                "response_draft", "diary", "identity", "trainer_instructions",
                "training_responses",
            ),
            CANONICAL_REGION_ORDER.map { it.wireName },
        )
        assertEquals(0, LOGICAL_REGION_IDS[LogicalRegion.CONVERSATION_HISTORY])
        assertEquals(12, LOGICAL_REGION_IDS[LogicalRegion.TRAINING_RESPONSES])
    }

    @Test
    fun `core writable regions are exactly scratch and response_draft`() {
        assertEquals(setOf(LogicalRegion.SCRATCH, LogicalRegion.RESPONSE_DRAFT), CORE_WRITABLE_REGIONS)
        // Sealed regions may not declare core-writable policy.
        assertFailsWith<IllegalArgumentException> {
            RegionState(name = LogicalRegion.IDENTITY, writePolicy = WritePolicy.CORE_WRITABLE)
        }
        // Default policy derives from the region.
        assertEquals(WritePolicy.CORE_WRITABLE, RegionState(LogicalRegion.SCRATCH).effectiveWritePolicy)
        assertEquals(WritePolicy.SEALED, RegionState(LogicalRegion.IDENTITY).effectiveWritePolicy)
    }

    @Test
    fun `empty snapshot has all thirteen regions and stable field id`() {
        val a = SharedFieldSnapshot.empty()
        val b = SharedFieldSnapshot.empty()
        assertEquals(13, a.regions.size)
        assertEquals(a.fieldId, b.fieldId)
        assertEquals("shared-field-v4", a.schema)
    }

    @Test
    fun `field id chains through parent`() {
        val base = SharedFieldSnapshot.empty()
        val delta = FieldDelta(
            baseFieldId = base.fieldId, baseTickId = base.tickId,
            authorCoreId = "tester", passId = "consolidated",
            operations = listOf(InsertText(LogicalRegion.SCRATCH, 0, "hello")),
        )
        val next = applyDelta(base, delta)
        assertEquals(base.fieldId, next.parentFieldId)
        assertEquals(base.tickId + 1, next.tickId)
        assertTrue(next.fieldId != base.fieldId)
        assertEquals("hello", next.region(LogicalRegion.SCRATCH).text)
    }

    @Test
    fun `mask policy does not change canonical field id`() {
        val plain = SharedFieldSnapshot.fromTexts(mapOf(LogicalRegion.SCRATCH to "abc"))
        val masked = SharedFieldSnapshot.create(
            regions = mapOf(
                LogicalRegion.SCRATCH to RegionState(
                    name = LogicalRegion.SCRATCH,
                    spans = RegionState.fromText(LogicalRegion.SCRATCH, "abc").spans,
                    maskPolicy = RegionMaskPolicy.tailPercent(50),
                )
            )
        )
        assertEquals(plain.fieldId, masked.fieldId)
    }
}
