package com.gliksbot.axonhome

import app.cash.turbine.test
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.sim.SimControlPlaneClient
import com.gliksbot.axonhome.sim.SimulationDriver
import com.gliksbot.axonhome.ui.home.HomeViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class HomeViewModelTest {

    @Before
    fun setMain() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
    }

    @After
    fun resetMain() {
        Dispatchers.resetMain()
    }

    @Test
    fun `home state maps live sim state`() = runTest {
        val engine = SimulationEngine()
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val driver = SimulationDriver(engine, scope)
        val vm = HomeViewModel(engine, driver)

        val initial = vm.state.value
        assertEquals(0L, initial.tick)
        assertEquals(4, initial.totalCores)

        driver.stepOnce()
        val after = vm.state.value
        assertEquals(1L, after.tick)
        assertEquals(engine.field.fieldId, after.fieldId)
        assertTrue(after.beatTick >= 1)
        // Soul continuity receipts accumulate from the tick passes.
        assertTrue(after.soulReceipts > 0)
    }

    @Test
    fun `paused organism renders paused, never fake online`() = runTest {
        val engine = SimulationEngine()
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val driver = SimulationDriver(engine, scope)
        val vm = HomeViewModel(engine, driver)

        engine.pauseOrganism()
        driver.stepOnce()
        assertEquals("paused", vm.state.value.organismStatus)
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class SimAdapterTest {

    @Test
    fun `client surfaces live sim data through the contract`() = runTest {
        val engine = SimulationEngine()
        engine.tick()
        val client = SimControlPlaneClient(engine)

        val health = client.health()
        assertEquals("ok", health.controlPlane.status)
        assertEquals(1L, health.heart?.tickSequence)

        val head = client.fieldHead()
        assertEquals(engine.field.fieldId, head.head.fieldId)

        val snapshot = client.fieldSnapshot(head.head.fieldId)
        assertEquals(13, snapshot.regions.size)

        val cores = client.cores()
        assertEquals(4, cores.cores.size)

        val soul = client.soul("sim-core-1")
        assertEquals(4, soul.layers.size)
    }

    @Test
    fun `event stream replays real sim events with seqs`() = runTest {
        val engine = SimulationEngine()
        engine.tick()
        val client = SimControlPlaneClient(engine)
        val flow = client.events(lastSeq = null)
        flow.test {
            val first = awaitItem()
            assertEquals(0L, first.seq)
            val second = awaitItem()
            assertEquals(1L, second.seq)
            cancelAndIgnoreRemainingEvents()
        }
    }
}
