package com.gliksbot.axonhome.ui.audit

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.gliksbot.axonhome.AppContainer
import com.gliksbot.axonhome.controlplane.AuditEntry
import com.gliksbot.axonhome.ui.components.FreshnessChip
import com.gliksbot.axonhome.controlplane.Freshness

/**
 * Operational audit log (arch §14). This is NOT the Engineer's Ledger —
 * ledger writes only happen via the governed repo script, never from the app.
 */
@Composable
fun AuditScreen(container: AppContainer) {
    var entries by remember { mutableStateOf<List<AuditEntry>>(emptyList()) }
    var freshness by remember { mutableStateOf(Freshness.LIVE) }

    LaunchedEffect(Unit) {
        // Refresh on entry; the log also grows as the sim processes actions.
        while (true) {
            val list = runCatching { container.client.audit(limit = 200) }.getOrNull()
            if (list != null) {
                entries = list.entries
                freshness = list.freshness
            }
            kotlinx.coroutines.delay(2_000)
        }
    }

    Column(Modifier.fillMaxSize().padding(horizontal = 12.dp)) {
        FreshnessChip(freshness, simulation = true, modifier = Modifier.padding(vertical = 6.dp))
        Text(
            "Operational audit (Control Plane-side mirror). Not the Engineer's Ledger.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        LazyColumn {
            items(entries) { entry ->
                ListItem(
                    headlineContent = {
                        Text("${entry.command} → ${entry.target ?: "—"}", fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodyMedium)
                    },
                    supportingContent = {
                        Text(
                            "${entry.who}@${entry.device} · ${entry.timestamp} · ${entry.result.name.lowercase()}" +
                                (entry.error?.let { " · error: $it" } ?: ""),
                            style = MaterialTheme.typography.bodySmall,
                        )
                    },
                )
                HorizontalDivider()
            }
        }
    }
}
