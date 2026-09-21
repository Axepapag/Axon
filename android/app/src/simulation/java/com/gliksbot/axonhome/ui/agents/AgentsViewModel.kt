package com.gliksbot.axonhome.ui.agents

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gliksbot.axonhome.controlplane.AgentList
import com.gliksbot.axonhome.controlplane.ControlPlaneClient
import com.gliksbot.axonhome.controlplane.Freshness
import com.gliksbot.axonhome.core.events.AxonEventPayload
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.data.AgentProfile
import com.gliksbot.axonhome.data.ProfileStore
import com.gliksbot.axonhome.sim.SimulationDriver
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class AgentsState(
    val freshness: Freshness = Freshness.LIVE,
    val roster: AgentList? = null,
    val recentMessages: List<String> = emptyList(),
    val profiles: List<AgentProfile> = emptyList(),
    val feedback: String? = null,
)

class AgentsViewModel(
    private val engine: SimulationEngine,
    private val client: ControlPlaneClient,
    private val profiles: ProfileStore?,
    private val driver: SimulationDriver,
    private val audit: (String, String, String, String, String) -> Unit,
) : ViewModel() {

    private val _state = MutableStateFlow(AgentsState())
    val state: StateFlow<AgentsState> = _state

    init {
        refresh()
        viewModelScope.launch { driver.tickCounter.collect { refresh() } }
    }

    fun refresh() {
        viewModelScope.launch {
            val roster = runCatching { client.agents() }.getOrNull()
            val messages = engine.eventLog.ofType("AgentMessageReceived")
                .takeLast(8)
                .mapNotNull {
                    (it.payload as? AxonEventPayload.AgentMessageReceived)?.let { p ->
                        "${p.agentId}: ${p.text}"
                    }
                }
                .reversed()
            val localProfiles = profiles?.let { runCatching { it.agents() }.getOrDefault(emptyList()) } ?: emptyList()
            _state.value = _state.value.copy(
                freshness = Freshness.LIVE,
                roster = roster,
                recentMessages = messages,
                profiles = localProfiles,
            )
        }
    }

    fun saveProfile(profile: AgentProfile) {
        viewModelScope.launch {
            profiles?.saveAgent(profile)
            audit("save_agent_profile", profile.id, "n/a", profile.displayName, "saved")
            _state.value = _state.value.copy(feedback = "Agent profile '${profile.displayName}' saved locally.")
            refresh()
        }
    }

    fun deleteProfile(id: String) {
        viewModelScope.launch {
            profiles?.deleteAgent(id)
            audit("delete_agent_profile", id, "saved", "absent", "deleted")
            refresh()
        }
    }
}
