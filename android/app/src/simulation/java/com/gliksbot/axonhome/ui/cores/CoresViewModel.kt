package com.gliksbot.axonhome.ui.cores

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gliksbot.axonhome.controlplane.ControlPlaneClient
import com.gliksbot.axonhome.controlplane.Freshness
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.sim.SimulationDriver
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class CoreListItemUi(
    val coreId: String,
    val status: String,
    val architectureId: String,
    val parameterGeneration: String,
    val soulId: String?,
    val soulGeneration: Long?,
    val consolidatorEligible: Boolean,
)

data class CoreDetailUi(
    val coreId: String,
    val status: String,
    val architectureId: String,
    val parameterGeneration: String,
    val consolidatorEligible: Boolean,
    val dModel: Int,
    val nHeads: Int,
    val nLayers: Int,
    val ffnDim: Int,
    val stateTokens: Int,
    val pageSize: Int,
    val soulId: String,
    val soulGeneration: Int,
    val soulParentId: String?,
    val soulLayers: List<Pair<String, Long>>,
    val soulReceipts: Int,
    val trainingRole: String,
)

data class CoresState(
    val freshness: Freshness = Freshness.LIVE,
    val cores: List<CoreListItemUi> = emptyList(),
    val detail: CoreDetailUi? = null,
    val actionFeedback: String? = null,
)

class CoresViewModel(
    private val engine: SimulationEngine,
    private val client: ControlPlaneClient,
    private val driver: SimulationDriver,
) : ViewModel() {

    private val _state = MutableStateFlow(CoresState())
    val state: StateFlow<CoresState> = _state

    init {
        refresh()
        viewModelScope.launch { driver.tickCounter.collect { refresh() } }
    }

    fun refresh() {
        val items = engine.cores.map { simCore ->
            val soul = engine.soulOf(simCore.descriptor.coreId)
            CoreListItemUi(
                coreId = simCore.descriptor.coreId,
                status = simCore.descriptor.status.name.lowercase(),
                architectureId = simCore.descriptor.architectureId,
                parameterGeneration = simCore.descriptor.parameterGeneration,
                soulId = soul.soulId,
                soulGeneration = soul.generation.toLong(),
                consolidatorEligible = simCore.consolidatorEligible,
            )
        }
        _state.value = _state.value.copy(freshness = Freshness.LIVE, cores = items)
    }

    fun loadDetail(coreId: String) {
        val simCore = engine.cores.firstOrNull { it.descriptor.coreId == coreId } ?: return
        val soul = engine.soulOf(coreId)
        val trainer = engine.trainer
        _state.value = _state.value.copy(
            detail = CoreDetailUi(
                coreId = coreId,
                status = simCore.descriptor.status.name.lowercase(),
                architectureId = simCore.descriptor.architectureId,
                parameterGeneration = simCore.descriptor.parameterGeneration,
                consolidatorEligible = simCore.consolidatorEligible,
                dModel = simCore.anatomy.dModel,
                nHeads = simCore.anatomy.nHeads,
                nLayers = simCore.anatomy.nLayers,
                ffnDim = simCore.anatomy.ffnDim,
                stateTokens = simCore.anatomy.stateTokens,
                pageSize = simCore.anatomy.pageSize,
                soulId = soul.soulId,
                soulGeneration = soul.generation,
                soulParentId = soul.parentSoulId,
                soulLayers = soul.layers.map { it.temperature.name.lowercase() to it.payloadBytes },
                soulReceipts = engine.soulReceiptsOf(coreId).size,
                trainingRole = if (trainer?.config?.coreId == coreId) {
                    "training candidate carrier (step ${trainer.globalStep})"
                } else {
                    "participant"
                },
            ),
        )
    }

    // ---- controls (wired where :core exposes a governed seam; else honest) ----

    /** Deactivation is not a governed sim seam — honest stub, no fake state change. */
    fun deactivateCore(coreId: String) {
        _state.value = _state.value.copy(
            actionFeedback = "Deactivate $coreId: Not implemented — no governed Control Plane seam in the simulator.",
        )
        container_audit("core_deactivate_stub", coreId)
    }

    fun pauseCore(coreId: String) {
        _state.value = _state.value.copy(
            actionFeedback = "Pause $coreId: Not implemented — per-core pause is designed, not built.",
        )
        container_audit("core_pause_stub", coreId)
    }

    fun checkpointCore(coreId: String) {
        _state.value = _state.value.copy(
            actionFeedback = "Checkpoint $coreId: Not implemented — checkpoints are written by the Trainer at accepted steps.",
        )
        container_audit("core_checkpoint_stub", coreId)
    }

    private fun container_audit(command: String, target: String) {
        engine.auditLog.record(
            who = "operator", device = "android-app", command = command, target = target,
            timestamp = kotlinx.datetime.Clock.System.now().toString(),
            previousState = "unchanged", requestedState = "n/a", result = "not-implemented",
        )
    }
}
