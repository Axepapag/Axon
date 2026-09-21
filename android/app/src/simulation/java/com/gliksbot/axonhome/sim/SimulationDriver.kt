package com.gliksbot.axonhome.sim

import com.gliksbot.axonhome.core.events.AxonEvent
import com.gliksbot.axonhome.core.domain.CandidateLifecycleStatus
import com.gliksbot.axonhome.core.sim.SimulationEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * Drives the deterministic simulator: advances heartbeats on a configurable
 * interval and re-publishes the sim event log as a hot flow. The UI's
 * heartbeat visualisation is driven ONLY by these real tick events — there is
 * no decorative fake animation (arch §10.A).
 */
class SimulationDriver(
    val engine: SimulationEngine,
    private val scope: CoroutineScope,
) {
    /** Real sim events (core envelope), in seq order. */
    private val _events = MutableSharedFlow<AxonEvent>(extraBufferCapacity = 512)
    val events: SharedFlow<AxonEvent> = _events

    /** Monotonic tick counter; increments once per completed engine tick. */
    private val _tickCounter = MutableStateFlow(0L)
    val tickCounter: StateFlow<Long> = _tickCounter

    /** Tick cadence in milliseconds (Settings-adjustable). */
    val tickIntervalMs = MutableStateFlow(2_000L)

    /** Auto-advance switch (Settings-adjustable). */
    val autoTick = MutableStateFlow(true)

    /** When true, a RUNNING trainer keeps stepping each tick and auto-renews
     *  exhausted tranches (the sim's default "continuous training" behaviour). */
    val autoTrain = MutableStateFlow(true)

    val running = MutableStateFlow(false)

    private var job: Job? = null
    private var lastEmittedSeq = -1L

    fun start() {
        if (job != null) return
        running.value = true
        job = scope.launch {
            // Flush any events appended before start().
            drain()
            while (true) {
                if (autoTick.value) {
                    stepOnce()
                }
                delay(tickIntervalMs.value)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
        running.value = false
    }

    /** Advance exactly one heartbeat (+ one training step when active). */
    fun stepOnce() {
        engine.tick()
        val trainer = engine.trainer
        if (trainer != null && autoTrain.value &&
            trainer.lifecycle == CandidateLifecycleStatus.RUNNING
        ) {
            engine.stepTraining()
        }
        _tickCounter.value = _tickCounter.value + 1
        drain()
    }

    /** Publish any event-log entries not yet emitted. */
    fun drain() {
        val fresh = engine.eventLog.replayFrom(lastEmittedSeq + 1)
        for (event in fresh) {
            _events.tryEmit(event)
            lastEmittedSeq = event.seq
        }
    }
}
