package com.gliksbot.axonhome.ui.agents

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliksbot.axonhome.data.AgentProfile
import com.gliksbot.axonhome.ui.components.Chip
import com.gliksbot.axonhome.ui.components.FreshnessChip
import com.gliksbot.axonhome.ui.components.InfoRow
import com.gliksbot.axonhome.ui.components.SectionCard

@Composable
fun AgentsScreen(viewModel: AgentsViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var editing by remember { mutableStateOf<AgentProfile?>(null) }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        FreshnessChip(state.freshness, simulation = true)

        SectionCard("Roundtable roster (sim)") {
            state.roster?.agents?.forEach { agent ->
                Column(Modifier.padding(vertical = 4.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(agent.name, fontFamily = FontFamily.Monospace, modifier = Modifier.weight(1f))
                        Chip(
                            if (agent.online) "online" else "offline",
                            if (agent.online) Color(0xFF2E7D32) else Color(0xFF9E9E9E),
                        )
                    }
                    InfoRow("Provider", agent.provider)
                    InfoRow("Model", agent.model ?: "—")
                    InfoRow("Active task", agent.activeTask ?: "—")
                    InfoRow("Capabilities", agent.capabilities.let { caps ->
                        buildList {
                            if (caps.supportsStreaming) add("streaming")
                            if (caps.supportsTools) add("tools")
                            if (caps.supportsSystemPrompt) add("system-prompt")
                            if (caps.supportsModelList) add("model-list")
                        }.joinToString().ifEmpty { "none" }
                    })
                }
                HorizontalDivider()
            } ?: Text("Roster unavailable.")
        }

        SectionCard("Recent Roundtable messages (sim)") {
            if (state.recentMessages.isEmpty()) Text("No messages yet — they arrive as the sim ticks.")
            state.recentMessages.forEach {
                Text(it, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(vertical = 2.dp))
            }
        }

        SectionCard("Agent profiles (local)") {
            state.profiles.forEach { profile ->
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 3.dp)) {
                    Column(Modifier.weight(1f)) {
                        Text(profile.displayName, style = MaterialTheme.typography.bodyMedium)
                        Text(
                            "${profile.provider} · ${profile.model.ifEmpty { "no model" }} · key: ${profile.vaultKeyRef ?: "none"}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    OutlinedButton(onClick = { editing = profile }) { Text("Edit") }
                    OutlinedButton(onClick = { viewModel.deleteProfile(profile.id) }, Modifier.padding(start = 4.dp)) { Text("Delete") }
                }
            }
            Button(onClick = {
                editing = AgentProfile(
                    id = "agent-${System.currentTimeMillis()}",
                    displayName = "",
                    provider = "openai-compatible",
                )
            }, modifier = Modifier.padding(top = 6.dp)) { Text("New agent profile") }
        }

        state.feedback?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
        }
    }

    editing?.let { profile ->
        AgentEditorDialog(
            initial = profile,
            onSave = { viewModel.saveProfile(it); editing = null },
            onDismiss = { editing = null },
        )
    }
}

@Composable
private fun AgentEditorDialog(
    initial: AgentProfile,
    onSave: (AgentProfile) -> Unit,
    onDismiss: () -> Unit,
) {
    var name by remember { mutableStateOf(initial.displayName) }
    var provider by remember { mutableStateOf(initial.provider) }
    var baseUrl by remember { mutableStateOf(initial.baseUrl) }
    var model by remember { mutableStateOf(initial.model) }
    var keyRef by remember { mutableStateOf(initial.vaultKeyRef ?: "") }
    var instructions by remember { mutableStateOf(initial.systemInstructions) }
    var allowCompute by remember { mutableStateOf(initial.allowCompute) }
    var allowStorage by remember { mutableStateOf(initial.allowStorage) }
    var allowRepo by remember { mutableStateOf(initial.allowRepo) }
    var timeout by remember { mutableStateOf(initial.timeoutSeconds.toString()) }

    androidx.compose.material3.AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Agent profile") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                OutlinedTextField(name, { name = it }, label = { Text("Display name") }, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(provider, { provider = it }, label = { Text("Provider") }, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(baseUrl, { baseUrl = it }, label = { Text("Base URL") }, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(model, { model = it }, label = { Text("Model") }, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(
                    keyRef, { keyRef = it },
                    label = { Text("API key vault REFERENCE (never the raw key)") },
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(instructions, { instructions = it }, label = { Text("System instructions") }, modifier = Modifier.fillMaxWidth())
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(allowCompute, { allowCompute = it }); Text("Compute")
                    Checkbox(allowStorage, { allowStorage = it }); Text("Storage")
                    Checkbox(allowRepo, { allowRepo = it }); Text("Repo")
                }
                OutlinedTextField(timeout, { timeout = it.filter(Char::isDigit) }, label = { Text("Timeout (s)") }, modifier = Modifier.fillMaxWidth())
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    onSave(
                        initial.copy(
                            displayName = name.trim().ifEmpty { "agent" },
                            provider = provider.trim(),
                            baseUrl = baseUrl.trim(),
                            model = model.trim(),
                            vaultKeyRef = keyRef.trim().ifEmpty { null },
                            systemInstructions = instructions,
                            allowCompute = allowCompute,
                            allowStorage = allowStorage,
                            allowRepo = allowRepo,
                            timeoutSeconds = timeout.toIntOrNull() ?: 60,
                        )
                    )
                },
                enabled = name.isNotBlank(),
            ) { Text("Save") }
        },
        dismissButton = { OutlinedButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
