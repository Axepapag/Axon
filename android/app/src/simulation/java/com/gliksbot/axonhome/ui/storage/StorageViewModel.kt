package com.gliksbot.axonhome.ui.storage

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gliksbot.axonhome.controlplane.Artifact
import com.gliksbot.axonhome.controlplane.CapsuleExportRequest
import com.gliksbot.axonhome.controlplane.CapsuleKind
import com.gliksbot.axonhome.controlplane.CapsuleSummary
import com.gliksbot.axonhome.controlplane.ControlPlaneClient
import com.gliksbot.axonhome.controlplane.Freshness
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.sim.SimulationDriver
import com.gliksbot.axonhome.ui.state.CapsuleVerifySummary
import com.gliksbot.axonhome.ui.state.summarizeCapsuleReport
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class StorageState(
    val freshness: Freshness = Freshness.LIVE,
    val artifacts: List<Artifact> = emptyList(),
    val capsules: List<CapsuleSummary> = emptyList(),
    val feedback: String? = null,
    val verifying: Boolean = false,
)

class StorageViewModel(
    private val engine: SimulationEngine,
    private val client: ControlPlaneClient,
    private val driver: SimulationDriver,
) : ViewModel() {

    private val _state = MutableStateFlow(StorageState())
    val state: StateFlow<StorageState> = _state

    private val _verifySummary = MutableStateFlow<CapsuleVerifySummary?>(null)
    val verifySummary: StateFlow<CapsuleVerifySummary?> = _verifySummary

    init {
        refresh()
        viewModelScope.launch { driver.tickCounter.collect { refresh() } }
    }

    fun refresh() {
        viewModelScope.launch {
            val artifacts = runCatching { client.artifacts().artifacts }.getOrDefault(emptyList())
            val capsules = runCatching { client.capsules().capsules }.getOrDefault(emptyList())
            _state.value = _state.value.copy(
                freshness = Freshness.LIVE,
                artifacts = artifacts,
                capsules = capsules,
            )
        }
    }

    /** Export a full-organism recovery capsule from the sim (fail-closed). */
    fun exportCapsule() {
        viewModelScope.launch {
            val result = runCatching {
                client.exportCapsule(
                    CapsuleExportRequest(kind = CapsuleKind.FULL_ORGANISM),
                    idempotencyKey = "capsule-${System.currentTimeMillis()}",
                )
            }
            result.onSuccess { manifest ->
                _state.value = _state.value.copy(
                    feedback = "Capsule exported: ${manifest.capsuleId.take(28)}… (${manifest.members.size} members)",
                )
                refresh()
            }.onFailure { error ->
                _state.value = _state.value.copy(
                    feedback = "Export refused: ${error.message} (need a candidate + checkpoint — start training first)",
                )
            }
        }
    }

    /** Fail-closed restore verification via :core's capsule verifier. */
    fun verifyCapsule(capsuleId: String) {
        viewModelScope.launch {
            _state.value = _state.value.copy(verifying = true)
            val result = runCatching {
                client.verifyCapsule(capsuleId, idempotencyKey = "verify-${System.currentTimeMillis()}")
            }
            _state.value = _state.value.copy(verifying = false)
            result.onSuccess { report -> _verifySummary.value = summarizeCapsuleReport(report) }
            result.onFailure { error ->
                _state.value = _state.value.copy(feedback = "Verify failed to run: ${error.message}")
            }
        }
    }

    fun dismissVerify() {
        _verifySummary.value = null
    }
}
