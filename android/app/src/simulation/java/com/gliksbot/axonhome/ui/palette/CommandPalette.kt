package com.gliksbot.axonhome.ui.palette

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.gliksbot.axonhome.AppContainer
import com.gliksbot.axonhome.ui.nav.Routes
import com.gliksbot.axonhome.ui.nav.moreEntries

private data class Command(
    val label: String,
    val keywords: String,
    val action: CommandAction,
)

private sealed class CommandAction {
    data class Navigate(val route: String) : CommandAction()
    data class Run(val run: (AppContainer) -> String) : CommandAction()
}

@Composable
fun CommandPalette(
    container: AppContainer,
    onDismiss: () -> Unit,
    onNavigate: (String) -> Unit,
) {
    var query by remember { mutableStateOf("") }
    var feedback by remember { mutableStateOf<String?>(null) }

    val commands = remember(container) {
        buildList {
            add(Command("Lock vault now", "lock secrets vault security", CommandAction.Run { c ->
                c.lockVault(); "Vault locked."
            }))
            add(Command("Step one tick", "tick heartbeat advance step", CommandAction.Run { c ->
                c.driver.stepOnce(); "Advanced one tick (now t${c.engine.field.tickId})."
            }))
            add(Command("Pause organism", "pause axon stop heartbeat", CommandAction.Run { c ->
                val r = c.engine.pauseOrganism(); "Paused at ${r.boundary}."
            }))
            add(Command("Resume organism", "resume axon start heartbeat", CommandAction.Run { c ->
                c.engine.resumeOrganism(); "Resumed."
            }))
            moreEntries.forEach { entry ->
                add(Command("Open ${entry.label}", entry.label.lowercase() + " " + entry.note.lowercase(), CommandAction.Navigate(entry.route)))
            }
            add(Command("Open Home", "home organism overview", CommandAction.Navigate(Routes.HOME)))
            add(Command("Open Field", "shared field explorer regions", CommandAction.Navigate(Routes.FIELD)))
            add(Command("Open Cores", "cores list observatory", CommandAction.Navigate(Routes.CORES)))
            add(Command("Open Trainer", "trainer training dashboard", CommandAction.Navigate(Routes.TRAINER)))
            add(Command("Open Audit log", "audit settings log", CommandAction.Navigate(Routes.AUDIT)))
            container.engine.cores.forEach { core ->
                add(Command(
                    "Open core ${core.descriptor.coreId}",
                    "core ${core.descriptor.coreId}",
                    CommandAction.Navigate(Routes.coreDetail(core.descriptor.coreId)),
                ))
            }
            container.engine.agents.forEach { agent ->
                add(Command("Agent $agent", "agent $agent roundtable", CommandAction.Navigate(Routes.AGENTS)))
            }
        }
    }

    val filtered = remember(query, commands) {
        if (query.isBlank()) commands
        else commands.filter {
            it.label.contains(query, ignoreCase = true) || it.keywords.contains(query, ignoreCase = true)
        }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Command palette") },
        text = {
            Column {
                OutlinedTextField(
                    value = query,
                    onValueChange = { query = it },
                    placeholder = { Text("Search commands, screens, cores, agents…") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                )
                feedback?.let {
                    Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
                }
                LazyColumn(Modifier.fillMaxWidth().heightIn(max = 360.dp).padding(top = 8.dp)) {
                    items(filtered) { command ->
                        ListItem(
                            headlineContent = { Text(command.label, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodyMedium) },
                            modifier = Modifier.clickable {
                                when (val action = command.action) {
                                    is CommandAction.Navigate -> onNavigate(action.route)
                                    is CommandAction.Run -> feedback = action.run(container)
                                }
                            },
                        )
                        HorizontalDivider()
                    }
                }
            }
        },
        confirmButton = { OutlinedButton(onClick = onDismiss) { Text("Close") } },
    )
}
