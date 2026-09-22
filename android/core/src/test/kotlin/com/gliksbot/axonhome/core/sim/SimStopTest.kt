package com.gliksbot.axonhome.core.sim

import com.gliksbot.axonhome.core.stop.ComponentAck
import com.gliksbot.axonhome.core.stop.StopKind
import com.gliksbot.axonhome.core.stop.StopResult
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SimStopTest {

    @Test
    fun `safe training stop preserves checkpoint soul pointer and tranche`() {
        val engine = SimulationEngine(seed = 53)
        engine.startTraining(SimTrainingConfig(trancheSteps = 8))
        repeat(3) { engine.stepTraining() }
        val result = engine.stopTraining(StopKind.SAFE)
        assertTrue(result is StopResult.Safe)
        val safe = (result as StopResult.Safe).result
        assertEquals("accepted-step:3", safe.boundary)
        assertTrue(safe.recoverable)
        assertEquals(4, safe.preserved.size) // checkpoint, pointer, soul, tranche
        assertTrue(safe.preserved.any { it.startsWith("checkpoint:") })
        // Training does not advance while paused.
        assertFalse(engine.stepTraining())
    }

    @Test
    fun `emergency training stop acks every component and never fakes lost workers`() {
        val engine = SimulationEngine(seed = 59)
        engine.startTraining(SimTrainingConfig(trancheSteps = 8))
        engine.stepTraining()
        engine.loseWorker("sim-worker-ssh-01", "link down")
        val result = engine.stopTraining(StopKind.EMERGENCY)
        assertTrue(result is StopResult.Emergency)
        val emergency = (result as StopResult.Emergency).result
        // The lost worker cannot acknowledge — must be UNREACHABLE, not STOPPED.
        assertEquals(ComponentAck.UNREACHABLE, emergency.acknowledgements["worker:sim-worker-ssh-01"])
        assertEquals(ComponentAck.STOPPED, emergency.acknowledgements["worker:sim-worker-kaggle-01"])
        assertFalse(emergency.allStopped)
    }

    @Test
    fun `emergency stop all covers heart cores trainer and workers`() {
        val engine = SimulationEngine(seed = 61)
        engine.startTraining(SimTrainingConfig(trancheSteps = 4))
        engine.stepTraining()
        val result = engine.emergencyStopAll()
        assertTrue(result.acknowledgements.keys.contains("heart"))
        assertEquals(4, result.acknowledgements.keys.count { it.startsWith("core:") })
        assertTrue(result.acknowledgements.keys.any { it.startsWith("trainer:") })
        assertTrue(result.acknowledgements.keys.count { it.startsWith("worker:") } == 2)
        assertTrue(result.allStopped) // all connected at this point
        // Ticks are suppressed after an emergency stop.
        kotlin.test.assertNull(engine.tick())
    }
}
