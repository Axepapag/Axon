package com.gliksbot.axonhome.ui.cores

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliksbot.axonhome.ui.components.Chip
import com.gliksbot.axonhome.ui.components.FreshnessChip
import com.gliksbot.axonhome.ui.components.InfoRow
import com.gliksbot.axonhome.ui.components.NotImplementedButton
import com.gliksbot.axonhome.ui.components.SectionCard
import com.gliksbot.axonhome.ui.components.TwoPane

@Composable
fun CoreListScreen(viewModel: CoresViewModel, expanded: Boolean, onOpenCore: (String) -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val listPane: @Composable () -> Unit = {
        LazyColumn(Modifier.fillMaxSize().padding(horizontal = 12.dp)) {
            item { FreshnessChip(state.freshness, simulation = true, modifier = Modifier.padding(vertical = 4.dp)) }
            items(state.cores) { core ->
                CoreCard(core, onClick = {
                    viewModel.loadDetail(core.coreId)
                    if (!expanded) onOpenCore(core.coreId)
                })
            }
        }
    }
    if (expanded) {
        val detailPane: @Composable () -> Unit = {
            val detail = state.detail
            if (detail == null) {
                Column(Modifier.fillMaxSize().padding(24.dp)) { Text("Select a core") }
            } else {
                CoreDetailContent(detail, state.actionFeedback, viewModel)
            }
        }
        TwoPane(expanded = true, first = listPane, second = detailPane)
    } else {
        listPane()
    }
}

@Composable
private fun CoreCard(core: CoreListItemUi, onClick: () -> Unit) {
    SectionCard(
        title = core.coreId,
        modifier = Modifier.padding(top = 8.dp).clickable(onClick = onClick),
    ) {
        Row {
            Chip(
                core.status,
                if (core.status == "active") Color(0xFF2E7D32) else Color(0xFF9E9E9E),
            )
            if (core.consolidatorEligible) Chip("consolidator", MaterialTheme.colorScheme.primary)
        }
        InfoRow("Architecture", core.architectureId.take(28) + "…", mono = true)
        InfoRow("Params", core.parameterGeneration, mono = true)
        InfoRow("Soul gen", core.soulGeneration?.toString() ?: "—", mono = true)
    }
}

@Composable
fun CoreDetailScreen(viewModel: CoresViewModel, coreId: String, onBack: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    LaunchedEffect(coreId) { viewModel.loadDetail(coreId) }
    val detail = state.detail
    Column(Modifier.fillMaxSize().padding(12.dp)) {
        OutlinedButton(onClick = onBack) { Text("← Back") }
        if (detail == null) {
            Text("Core not found", Modifier.padding(top = 12.dp))
        } else {
            CoreDetailContent(detail, state.actionFeedback, viewModel)
        }
    }
}

@Composable
private fun CoreDetailContent(detail: CoreDetailUi, feedback: String?, viewModel: CoresViewModel) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(detail.coreId, style = MaterialTheme.typography.titleLarge, fontFamily = FontFamily.Monospace)
            Chip(
                detail.status,
                if (detail.status == "active") Color(0xFF2E7D32) else Color(0xFF9E9E9E),
                Modifier.padding(start = 8.dp),
            )
        }

        SectionCard("Anatomy (D64-locked)", Modifier.padding(top = 10.dp)) {
            InfoRow("architecture_id", detail.architectureId, mono = true)
            InfoRow("d_model", detail.dModel.toString(), mono = true)
            InfoRow("n_heads", detail.nHeads.toString(), mono = true)
            InfoRow("n_layers", detail.nLayers.toString(), mono = true)
            InfoRow("ffn_dim", detail.ffnDim.toString(), mono = true)
            InfoRow("state_tokens", detail.stateTokens.toString(), mono = true)
            InfoRow("page_size", detail.pageSize.toString(), mono = true)
            InfoRow("parameter_generation", detail.parameterGeneration, mono = true)
            InfoRow("role", if (detail.consolidatorEligible) "participant + consolidator" else "participant")
            InfoRow("training", detail.trainingRole)
        }

        SectionCard("Soul", Modifier.padding(top = 10.dp)) {
            InfoRow("soul_id", detail.soulId.take(32) + "…", mono = true)
            InfoRow("generation", detail.soulGeneration.toString(), mono = true)
            InfoRow("parent", detail.soulParentId?.take(24)?.plus("…") ?: "genesis", mono = true)
            InfoRow("continuity receipts", detail.soulReceipts.toString(), mono = true)
            detail.soulLayers.forEach { (temp, bytes) ->
                InfoRow("layer $temp", "$bytes bytes (opaque)", mono = true)
            }
            Text(
                "Latent tensors are opaque and are never rendered as semantics.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        SectionCard("Controls", Modifier.padding(top = 10.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Button(onClick = { viewModel.pauseCore(detail.coreId) }) { Text("Pause") }
                OutlinedButton(onClick = { viewModel.deactivateCore(detail.coreId) }) { Text("Deactivate") }
                OutlinedButton(onClick = { viewModel.checkpointCore(detail.coreId) }) { Text("Checkpoint") }
            }
            feedback?.let {
                Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.tertiary)
            }
        }
    }
}
