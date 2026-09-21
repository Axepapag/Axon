package com.gliksbot.axonhome.core.phone

import com.gliksbot.axonhome.core.domain.*
import kotlin.test.*

class PhoneStateTest {
    private fun initial() = PhoneState(field = SharedFieldSnapshot.empty())
    @Test fun pythonCanonicalFieldFixtureMatches() {
        val field = SharedFieldSnapshot.create(regions = mapOf(LogicalRegion.USER_INPUT to
            RegionState.fromText(LogicalRegion.USER_INPUT, "a🧠z", spanId = "input-1", source = "android-user", provenance = "input-1")))
        assertEquals("3923860fb93bcbfe8f489d62258d6f5296a6abe4b537a01c82dc2252961b1b34", field.fieldId)
    }
    @Test fun ingressPreservesExactTextAndDoesNotInventReasoning() {
        val before = initial()
        val after = PhoneHeart.ingress(before, "Hello 🧠\n世界", "input-1")
        assertEquals("Hello 🧠\n世界", after.field.region(LogicalRegion.USER_INPUT).text)
        assertEquals(before.field.fieldId, after.field.parentFieldId)
        assertEquals(0, after.field.tickId)
        assertEquals(1L, after.ingressCount)
        assertEquals("", before.field.region(LogicalRegion.USER_INPUT).text)
        assertEquals("", after.field.region(LogicalRegion.RESPONSE_DRAFT).text)
    }
    @Test fun masksAreScalarBasedAndDoNotChangeFieldIdentity() {
        val state = PhoneHeart.ingress(initial(), "a🧠z", "input-1")
        val masked = PhoneHeart.mask(state, LogicalRegion.USER_INPUT, 50)
        assertEquals(state.field.fieldId, masked.field.fieldId)
        val view = masked.field.region(LogicalRegion.USER_INPUT).copy(maskPolicy = RegionMaskPolicy.tailPercent(50))
        assertEquals(listOf(AttendedInterval(1, 3)), view.resolvedAttendedIntervals)
        assertEquals("🧠z", view.attendedText)
        assertFails { PhoneHeart.mask(state, LogicalRegion.IDENTITY, 0) }
    }
    @Test fun pauseAndInvalidUnicodeFailWithoutMutation() {
        val state = initial().copy(paused = true)
        assertFails { PhoneHeart.ingress(state, "hello", "one") }
        assertFails { PhoneHeart.ingress(initial(), "\uD800", "two") }
        assertEquals(0L, state.ingressCount)
    }
    @Test fun recoveryRoundTripsAndRejectsTamperingOrWrongSchema() {
        val state = PhoneHeart.ingress(initial(), "🧠\nhello", "input-1")
        val backup = PhoneRecovery.create(state, "source-commit")
        assertEquals(state, backup.verify())
        assertFails { backup.copy(payload = backup.payload + " ").verify() }
        assertFails { backup.copy(complete = false).verify() }
        assertFails { backup.copy(schema = "other").verify() }
    }
    @Test fun persistedStateRejectsMissingMasksAndIdentityMask() {
        assertFails { initial().copy(masks = emptyMap()) }
        assertFails { initial().copy(masks = initial().masks + ("identity" to 0)) }
    }
}
