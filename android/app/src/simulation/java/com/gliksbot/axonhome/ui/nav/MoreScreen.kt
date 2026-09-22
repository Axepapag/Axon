package com.gliksbot.axonhome.ui.nav

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.gliksbot.axonhome.AppContainer
import com.gliksbot.axonhome.ui.components.Chip
import com.gliksbot.axonhome.ui.components.InfoRow
import com.gliksbot.axonhome.ui.components.SectionCard
import com.gliksbot.axonhome.ui.theme.SimAccent

/** All destinations reachable from the More screen (arch §10 A–T coverage). */
data class MoreEntry(val label: String, val route: String, val note: String)

val moreEntries = listOf(
    MoreEntry("Compute Fleet", Routes.COMPUTE, "Workers, profiles, capabilities"),
    MoreEntry("Storage Center", Routes.STORAGE, "Artifacts, capsules, verify"),
    MoreEntry("Engineers / Agents", Routes.AGENTS, "Roundtable roster + agent profiles"),
    MoreEntry("Attention / Region Masks", Routes.MASKS, "Two-step confirm mask control"),
    MoreEntry("Secrets Vault", Routes.VAULT, "Keystore-sealed secrets (foundation)"),
    MoreEntry("Settings", Routes.SETTINGS, "Endpoints, theme, security, about"),
    MoreEntry("Tick Time Machine", Routes.TIME_MACHINE, "Stub — Phase 6"),
    MoreEntry("Rail Observatory", Routes.RAIL_OBSERVATORY, "Stub — Phase 6"),
    MoreEntry("Semantic Cortex", Routes.SEMANTIC_CORTEX, "Stub — Phase 6"),
    MoreEntry("Soul Observatory (deep)", Routes.SOUL_OBSERVATORY, "Stub — Phase 6"),
    MoreEntry("Dormant", Routes.DORMANT, "Stats card; browser stub"),
    MoreEntry("Notebook / Remote Workspace", Routes.NOTEBOOK, "Stub — Phase 4"),
    MoreEntry("SSH Terminal", Routes.SSH_TERMINAL, "Stub — Phase 4"),
    MoreEntry("Git / Repository", Routes.GIT_CLIENT, "Status card; ops stub"),
)

@Composable
fun MoreScreen(onNavigate: (String) -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(horizontal = 12.dp)) {
        items(moreEntries) { entry ->
            ListItem(
                headlineContent = { Text(entry.label) },
                supportingContent = { Text(entry.note, style = MaterialTheme.typography.bodySmall) },
                modifier = Modifier.clickable { onNavigate(entry.route) },
            )
            HorizontalDivider()
        }
    }
}

/** Dormant: real sim stats card; the corpus browser is an honest stub (Phase 6). */
@Composable
fun DormantScreen(container: AppContainer) {
    val engine = container.engine
    Column(Modifier.fillMaxSize().padding(12.dp)) {
        SectionCard("Dormant corpus — sim stats") {
            InfoRow("Dormant index", "sim-dormant-index", mono = true)
            InfoRow("Corpus sources", "0 imported (simulation)")
            InfoRow("Curriculum-eligible", "0")
            InfoRow("Quarantined", "0")
            InfoRow("Storage used", "0 B")
        }
        SectionCard("Dormant browser", Modifier.padding(top = 12.dp)) {
            Text(
                "Not implemented yet — Phase 6 (arch §18). Corpus provenance, grouping, " +
                    "ingestion state and search require the dormant evidence index, which " +
                    "the simulator does not emit.",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

/** Git / Repository: real status card from sim build metadata; operations stubbed. */
@Composable
fun GitScreen(container: AppContainer) {
    Column(Modifier.fillMaxSize().padding(12.dp)) {
        SectionCard("Repository status") {
            InfoRow("Branch", "main (sim)")
            InfoRow("Commit", com.gliksbot.axonhome.BuildConfig.GIT_COMMIT, mono = true)
            InfoRow("Version", com.gliksbot.axonhome.BuildConfig.VERSION_NAME)
            InfoRow("Ahead / behind", "0 / 0 (sim)")
        }
        SectionCard("Operations", Modifier.padding(top = 12.dp)) {
            Text(
                "Fetch/pull/commit/push are not implemented in this slice (no network, " +
                    "no GitProvider adapter). Destructive operations will always confirm " +
                    "first; force-push will never be offered.",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}
