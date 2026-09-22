package com.gliksbot.axonhome.ui.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gliksbot.axonhome.BuildConfig
import com.gliksbot.axonhome.controlplane.Freshness
import com.gliksbot.axonhome.core.events.AxonEventPayload
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.sim.SimulationDriver
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class HomeState(
    val freshness: Freshness = Freshness.LIVE,
    val organismStatus: String = "starting",
    val tick: Long = 0,
    val heartbeatSequence: Long = 0,
    val fieldId: String = "",
    val activeCores: Int = 0,
    val totalCores: Int = 0,
    val consolidatorEligible: Int = 0,
    val trainerLifecycle: String = "no candidate",
    val trainerStep: Long = 0,
    val latestLoss: Double? = null,
    val lineage: String = "",
    val latestCheckpoint: String? = null,
    val soulReceipts: Int = 0,
    val workersConnected: Int = 0,
    val workersTotal: Int = 0,
    val agentsOnline: Int = 0,
    val alerts: List<String> = emptyList(),
    val lastCapsule: String? = null,
    val gitCommit: String = "",
    val versionName: String = "",
    val beatTick: Long = 0,
)

class HomeViewModel(
    private val engine: SimulationEngine,
    private val driver: SimulationDriver,
) : ViewModel() {

    private val _state = MutableStateFlow(HomeState())
    val state: StateFlow<HomeState> = _state

    init {
        refresh()
        viewModelScope.launch {
            driver.tickCounter.collect { refresh() }
        }
    }

    fun refresh() {
        val field = engine.field
        val health = engine.health()
        val trainer = engine.trainer
        val alerts = engine.eventLog.ofType("AlertRaised")
            .takeLast(5)
            .mapNotNull { (it.payload as? AxonEventPayload.AlertRaised)?.let { p -> "${p.severity}: ${p.message}" } }
            .reversed()
        val capsules = engine.eventLog.ofType("RecoveryCapsulePublished")
            .mapNotNull { (it.payload as? AxonEventPayload.RecoveryCapsulePublished)?.capsuleId }
        _state.value = HomeState(
            freshness = Freshness.LIVE,
            organismStatus = if (engine.organismPaused) "paused" else "online",
            tick = field.tickId.toLong(),
            heartbeatSequence = health.heartbeatSequence,
            fieldId = field.fieldId,
            activeCores = engine.activeCores.size,
            totalCores = engine.cores.size,
            consolidatorEligible = engine.cores.count { it.consolidatorEligible },
            trainerLifecycle = trainer?.lifecycle?.name ?: "no candidate",
            trainerStep = trainer?.globalStep?.toLong() ?: 0,
            latestLoss = trainer?.lastLoss?.takeIf { !it.isNaN() },
            lineage = if (trainer != null) {
                "${trainer.baseGenerationId.take(20)}… → ${trainer.candidateGenerationId.take(20)}…"
            } else {
                "no lineage yet"
            },
            latestCheckpoint = trainer?.latestCheckpoint?.checkpointId,
            soulReceipts = engine.cores.sumOf { engine.soulReceiptsOf(it.descriptor.coreId).size },
            workersConnected = engine.workers.count { it.connected },
            workersTotal = engine.workers.size,
            agentsOnline = engine.agents.size,
            alerts = alerts,
            lastCapsule = capsules.lastOrNull(),
            gitCommit = BuildConfig.GIT_COMMIT,
            versionName = BuildConfig.VERSION_NAME,
            beatTick = health.heartbeatSequence,
        )
    }
}
