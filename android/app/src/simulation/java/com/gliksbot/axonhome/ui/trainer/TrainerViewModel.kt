package com.gliksbot.axonhome.ui.trainer

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gliksbot.axonhome.controlplane.ControlPlaneClient
import com.gliksbot.axonhome.controlplane.Freshness
import com.gliksbot.axonhome.controlplane.StopRequest
import com.gliksbot.axonhome.controlplane.StopResponse
import com.gliksbot.axonhome.controlplane.StopScope
import com.gliksbot.axonhome.controlplane.TrainerCommandKind
import com.gliksbot.axonhome.controlplane.TrainerCommandRequest
import com.gliksbot.axonhome.controlplane.TrainerCommandStatus
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.sim.SimulationDriver
import com.gliksbot.axonhome.ui.state.StopSummary
import com.gliksbot.axonhome.ui.state.summarizeStop
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class TrainerState(
    val freshness: Freshness = Freshness.LIVE,
    val lifecycle: String = "no candidate",
    val moduleId: String = "",
    val baseGeneration: String = "",
    val candidateGeneration: String = "",
    val globalStep: Int = 0,
    val trancheBase: Int = 0,
    val trancheFinal: Int = 0,
    val tranchePurpose: String = "",
    val optimizer: String = "",
    val learningRate: Double = 0.0,
    val losses: List<Double> = emptyList(),
    val gradNorms: List<Double> = emptyList(),
    val retainedCheckpoints: List<String> = emptyList(),
    val heldoutEvals: Int = 0,
    val workers: List<Pair<String, Boolean>> = emptyList(),
    val commandFeedback: String? = null,
    val availableCommands: Map<TrainerCommandKind, Boolean> = emptyMap(),
)

class TrainerViewModel(
    private val engine: SimulationEngine,
    private val client: ControlPlaneClient,
    private val driver: SimulationDriver,
) : ViewModel() {

    private val _state = MutableStateFlow(TrainerState())
    val state: StateFlow<TrainerState> = _state

    /** Result of the last stop action, rendered truthfully (arch §13). */
    private val _stopSummary = MutableStateFlow<StopSummary?>(null)
    val stopSummary: StateFlow<StopSummary?> = _stopSummary

    private var idempotencyCounter = 0L
    private fun nextKey() = "app-${System.currentTimeMillis()}-${idempotencyCounter++}"

    init {
        refresh()
        viewModelScope.launch { driver.tickCounter.collect { refresh() } }
    }

    fun refresh() {
        val trainer = engine.trainer
        _state.value = _state.value.copy(
            freshness = Freshness.LIVE,
            lifecycle = trainer?.lifecycle?.name ?: "no candidate",
            moduleId = trainer?.config?.moduleId ?: "",
            baseGeneration = trainer?.baseGenerationId ?: "",
            candidateGeneration = trainer?.candidateGenerationId ?: "",
            globalStep = trainer?.globalStep ?: 0,
            trancheBase = trainer?.tranche?.baseGlobalStep ?: 0,
            trancheFinal = trainer?.tranche?.finalGlobalStep ?: 0,
            tranchePurpose = trainer?.tranche?.purpose ?: "",
            optimizer = trainer?.policy?.optimizer?.name ?: "",
            learningRate = trainer?.policy?.learningRate ?: 0.0,
            losses = trainer?.receipts?.map { it.loss }?.takeLast(60) ?: emptyList(),
            gradNorms = trainer?.receipts?.map { it.gradientL2 }?.takeLast(60) ?: emptyList(),
            retainedCheckpoints = trainer?.retainedCheckpoints?.map { "${it.checkpointId.take(20)}… @step ${it.step}" }
                ?: emptyList(),
            heldoutEvals = trainer?.heldoutEvaluations?.size ?: 0,
            workers = engine.workers.map { it.workerId to it.connected },
        )
    }

    fun command(kind: TrainerCommandKind) {
        viewModelScope.launch {
            val response = runCatching {
                client.trainerCommand(
                    TrainerCommandRequest(
                        kind = kind,
                        requestedBy = "operator",
                        idempotencyKey = nextKey(),
                    )
                )
            }
            response.onSuccess { result ->
                val prefix = when (result.status) {
                    TrainerCommandStatus.OK -> "OK"
                    TrainerCommandStatus.UNAVAILABLE -> "Not implemented (governed UNAVAILABLE)"
                    TrainerCommandStatus.REJECTED -> "Rejected"
                    TrainerCommandStatus.FAILED -> "Failed"
                }
                _state.value = _state.value.copy(commandFeedback = "$prefix: ${result.message}")
                refresh()
            }.onFailure { error ->
                _state.value = _state.value.copy(commandFeedback = "Failed: ${error.message}")
            }
        }
    }

    /** Red STOP: safe (boundary) or emergency (immediate, honest acks). */
    fun stop(scope: StopScope) {
        viewModelScope.launch {
            val response: StopResponse? = runCatching {
                client.stop(StopRequest(scope = scope), idempotencyKey = nextKey())
            }.getOrNull()
            if (response != null) {
                _stopSummary.value = summarizeStop(response)
                refresh()
            } else {
                _state.value = _state.value.copy(commandFeedback = "Stop request failed to reach the control plane (sim)")
            }
        }
    }

    fun dismissStopSummary() {
        _stopSummary.value = null
    }
}
