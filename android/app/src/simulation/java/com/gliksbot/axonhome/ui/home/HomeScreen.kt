package com.gliksbot.axonhome.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliksbot.axonhome.ui.components.Chip
import com.gliksbot.axonhome.ui.components.FreshnessChip
import com.gliksbot.axonhome.ui.components.HeartbeatPulse
import com.gliksbot.axonhome.ui.components.InfoRow
import com.gliksbot.axonhome.ui.components.SectionCard
import com.gliksbot.axonhome.ui.nav.Routes

@Composable
fun HomeScreen(viewModel: HomeViewModel, onNavigate: (String) -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
            FreshnessChip(state.freshness, simulation = true)
            Chip(
                text = state.organismStatus,
                color = if (state.organismStatus == "online") Color(0xFF2E7D32) else Color(0xFFEF6C00),
            )
        }

        SectionCard("Heartbeat") {
            // Pulse fires only on real tick events (beatTick), never decorative.
            HeartbeatPulse(beatTick = state.beatTick)
            InfoRow("Heartbeat sequence", state.heartbeatSequence.toString(), mono = true)
            InfoRow("Current tick", state.tick.toString(), mono = true)
            InfoRow("Heart status", if (state.organismStatus == "online") "beating" else "paused")
        }

        SectionCard("Canonical field", onClick = { onNavigate(Routes.FIELD) }) {
            InfoRow("Field id", state.fieldId.take(24) + "…", mono = true)
            InfoRow("Hash (short)", state.fieldId.take(12), mono = true)
        }

        SectionCard("Cores", onClick = { onNavigate(Routes.CORES) }) {
            InfoRow("Active cores", "${state.activeCores} / ${state.totalCores}")
            InfoRow("Consolidator-eligible", state.consolidatorEligible.toString())
            InfoRow("Dormant cores", (state.totalCores - state.activeCores).toString())
        }

        SectionCard("Trainer", onClick = { onNavigate(Routes.TRAINER) }) {
            InfoRow("Lifecycle", state.trainerLifecycle)
            InfoRow("Accepted step", state.trainerStep.toString(), mono = true)
            InfoRow("Latest loss", state.latestLoss?.let { "%.4f".format(it) } ?: "—", mono = true)
            InfoRow("Lineage (base→candidate)", state.lineage, mono = true)
            InfoRow("Latest checkpoint", state.latestCheckpoint?.take(24)?.plus("…") ?: "—", mono = true)
        }

        SectionCard("Souls") {
            InfoRow("Continuity receipts (all cores)", state.soulReceipts.toString(), mono = true)
        }

        SectionCard("Fleet & agents", onClick = { onNavigate(Routes.COMPUTE) }) {
            InfoRow("Compute connected", "${state.workersConnected} / ${state.workersTotal}")
            InfoRow("Agents online", state.agentsOnline.toString())
            InfoRow("Storage", "local mirror (sim)")
        }

        SectionCard("Recovery") {
            InfoRow("Last capsule", state.lastCapsule?.take(28)?.plus("…") ?: "none exported yet", mono = true)
        }

        if (state.alerts.isNotEmpty()) {
            SectionCard("Alerts") {
                state.alerts.forEach { alert ->
                    Text(
                        alert,
                        style = MaterialTheme.typography.bodySmall,
                        color = if (alert.startsWith("critical")) MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 2.dp),
                    )
                }
            }
        }

        SectionCard("Build (sim)") {
            InfoRow("Git branch", "main (sim)")
            InfoRow("Git commit", state.gitCommit, mono = true)
            InfoRow("App version", state.versionName)
        }
    }
}
