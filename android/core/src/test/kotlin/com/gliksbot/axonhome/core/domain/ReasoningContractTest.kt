package com.gliksbot.axonhome.core.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ReasoningContractTest {

    private val base: SharedFieldSnapshot = SharedFieldSnapshot.fromTexts(
        mapOf(LogicalRegion.SCRATCH to "old scratch", LogicalRegion.RESPONSE_DRAFT to "old draft")
    )

    @Test
    fun `region tags mirror the canonical surface including journal and diary alias`() {
        assertEquals("responseDraft", REGION_TAGS[LogicalRegion.RESPONSE_DRAFT])
        assertEquals("scratch", REGION_TAGS[LogicalRegion.SCRATCH])
        assertEquals("journal", REGION_TAGS[LogicalRegion.DIARY])
        assertEquals("conversationHistory", REGION_TAGS[LogicalRegion.CONVERSATION_HISTORY])
        assertEquals(LogicalRegion.DIARY, TAG_TO_REGION["journal"])
        assertEquals(LogicalRegion.DIARY, TAG_TO_REGION["diary"]) // accepted compatibility alias
        assertEquals(13, REGION_TAGS.size)
    }

    @Test
    fun `render then parse roundtrips tagged sections`() {
        val sections = listOf(
            LogicalRegion.RESPONSE_DRAFT to "first line\nsecond line",
            LogicalRegion.SCRATCH to "notes",
        )
        val text = renderFinalVerdict(sections)
        assertEquals("#responseDraft# first line\nsecond line\n#scratch# notes", text)
        assertEquals(sections, parseFinalVerdictSections(text))
    }

    @Test
    fun `lines that look like tags inside a body are escaped and unescaped`() {
        val sections = listOf(LogicalRegion.SCRATCH to "top\n#notATagButLooks# sneaky")
        val text = renderFinalVerdict(sections)
        assertTrue(text.contains("\n\\#notATagButLooks# sneaky"))
        assertEquals(sections, parseFinalVerdictSections(text))
    }

    @Test
    fun `verdict rejects unknown tags and duplicate sections`() {
        assertFailsWith<EnglishReasoningContractError> {
            TechnicalFinalVerdict(base.fieldId, base.tickId, "c", 64, "#bogus# hi")
        }
        assertFailsWith<EnglishReasoningContractError> {
            TechnicalFinalVerdict(base.fieldId, base.tickId, "c", 64, "#scratch# a\n#scratch# b")
        }
    }

    @Test
    fun `verdict materializes to a core writable delta only`() {
        val verdict = TechnicalFinalVerdict(
            baseFieldId = base.fieldId,
            baseTickId = base.tickId,
            authorCoreId = "sim-core-1",
            railDModel = 64,
            text = "#responseDraft# new answer\n#scratch# new notes",
        )
        val delta = verdict.materialize(base)
        assertEquals("consolidated", delta.passId)
        assertEquals(setOf(LogicalRegion.RESPONSE_DRAFT, LogicalRegion.SCRATCH),
            delta.operations.map { it.region }.toSet())
        val next = applyDelta(base, delta)
        assertEquals("new answer", next.region(LogicalRegion.RESPONSE_DRAFT).text)
        assertEquals("new notes", next.region(LogicalRegion.SCRATCH).text)
    }

    @Test
    fun `stale verdict materialization is rejected`() {
        val verdict = TechnicalFinalVerdict(base.fieldId, base.tickId, "sim-core-1", 64, "#scratch# n")
        val moved = applyDelta(base, verdict.materialize(base))
        assertFailsWith<EnglishReasoningContractError> { verdict.materialize(moved) }
    }

    @Test
    fun `proposal contract enforces pass id and rail width`() {
        assertFailsWith<EnglishReasoningContractError> {
            EnglishProposal(base.fieldId, base.tickId, "c", "consolidated", 64, "text")
        }
        assertFailsWith<EnglishReasoningContractError> {
            EnglishProposal(base.fieldId, base.tickId, "c", "first", 63, "text")
        }
        val ok = EnglishProposal(base.fieldId, base.tickId, "c", "first", 64, "text")
        assertEquals(64, ok.proposalId.length)
    }
}
