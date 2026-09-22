package com.gliksbot.axonhome.ui.compute

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gliksbot.axonhome.controlplane.ControlPlaneClient
import com.gliksbot.axonhome.controlplane.Freshness
import com.gliksbot.axonhome.controlplane.Worker
import com.gliksbot.axonhome.data.EndpointProfile
import com.gliksbot.axonhome.data.ProfileStore
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.sim.SimulationDriver
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class ComputeState(
    val freshness: Freshness = Freshness.LIVE,
    val workers: List<Worker> = emptyList(),
    val endpoints: List<EndpointProfile> = emptyList(),
    val feedback: String? = null,
    /** When set, a billable action awaits cost confirmation (arch §8). */
    val pendingBillableAction: String? = null,
)

class ComputeViewModel(
    private val engine: SimulationEngine,
    private val client: ControlPlaneClient,
    private val profiles: ProfileStore?,
    private val driver: SimulationDriver,
    private val audit: (String, String, String, String, String) -> Unit,
) : ViewModel() {

    private val _state = MutableStateFlow(ComputeState())
    val state: StateFlow<ComputeState> = _state

    init {
        refresh()
        viewModelScope.launch { driver.tickCounter.collect { refresh() } }
    }

    fun refresh() {
        viewModelScope.launch {
            val workers = runCatching { client.workers().workers }.getOrDefault(emptyList())
            val endpoints = profiles?.let { runCatching { it.endpoints() }.getOrDefault(emptyList()) } ?: emptyList()
            _state.value = _state.value.copy(
                freshness = Freshness.LIVE,
                workers = workers,
                endpoints = endpoints,
            )
        }
    }

    fun saveEndpoint(profile: EndpointProfile) {
        viewModelScope.launch {
            profiles?.saveEndpoint(profile)
            audit("save_endpoint_profile", profile.id, "n/a", profile.baseUrl, "saved")
            _state.value = _state.value.copy(feedback = "Endpoint profile '${profile.name}' saved locally.")
            refresh()
        }
    }

    fun deleteEndpoint(id: String) {
        viewModelScope.launch {
            profiles?.deleteEndpoint(id)
            audit("delete_endpoint_profile", id, "saved", "absent", "deleted")
            refresh()
        }
    }

    /** Billable actions always pass through the cost-confirmation dialog. */
    fun requestBillableAction(description: String) {
        _state.value = _state.value.copy(pendingBillableAction = description)
    }

    fun cancelBillableAction() {
        _state.value = _state.value.copy(pendingBillableAction = null)
    }

    fun confirmBillableAction() {
        val action = _state.value.pendingBillableAction ?: return
        audit("billable_action_stub", "compute", "n/a", action, "not-implemented")
        _state.value = _state.value.copy(
            pendingBillableAction = null,
            feedback = "Confirmed, but NOT executed: '$action' is not implemented in this slice (no provider adapters).",
        )
    }

    fun stopWorker(workerId: String) {
        viewModelScope.launch {
            val response = runCatching { client.stopWorker(workerId, "stop-${System.currentTimeMillis()}") }
            response.onSuccess {
                _state.value = _state.value.copy(
                    feedback = "Worker $workerId: ${it.ack.status.name.lowercase().replace('_', ' ')}" +
                        (it.ack.detail?.let { d -> " — $d" } ?: ""),
                )
            }
            refresh()
        }
    }
}
