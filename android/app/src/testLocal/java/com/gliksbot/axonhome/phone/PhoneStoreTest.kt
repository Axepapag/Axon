package com.gliksbot.axonhome.phone

import android.content.Context
import com.gliksbot.axonhome.core.domain.*
import com.gliksbot.axonhome.core.phone.*
import org.junit.*
import org.junit.Assert.*
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class PhoneStoreTest {
    private lateinit var context: Context
    private lateinit var store: PhoneStore
    @Before fun setup() {
        context = RuntimeEnvironment.getApplication()
        context.deleteDatabase("axon-phone.db")
        store = PhoneStore(context)
    }
    @After fun close() { store.close() }
    @Test fun reopenPreservesExactInputIdentityAndMask() {
        store.ingress("hello 🧠\n世界")
        val before = store.mask(LogicalRegion.USER_INPUT, 50)
        store.close()
        store = PhoneStore(context)
        assertEquals(before, store.state())
        assertEquals(2064, store.state().field.region(LogicalRegion.IDENTITY).text.length)
        assertEquals(0, store.state().field.tickId)
    }
    @Test fun transactionFailureRollsBackCanonicalBody() {
        val before = store.state()
        store.writableDatabase.execSQL("CREATE TRIGGER fail_audit BEFORE INSERT ON history BEGIN SELECT RAISE(ABORT, 'injected storage failure'); END")
        try { store.ingress("must roll back"); fail("expected failure") } catch (_: android.database.SQLException) {}
        assertEquals(before, store.state())
    }
    @Test fun tamperedRecoveryCannotChangeBodyAndValidRestoreSurvivesReopen() {
        val old = store.ingress("first")
        val backup = PhoneRecovery.create(old, "test-source")
        val newer = store.ingress("second")
        try { store.restore(backup.copy(sha256 = "bad")); fail("expected rejection") } catch (_: IllegalArgumentException) {}
        assertEquals(newer, store.state())
        store.restore(backup)
        store.close()
        store = PhoneStore(context)
        assertEquals(old, store.state())
    }
}
