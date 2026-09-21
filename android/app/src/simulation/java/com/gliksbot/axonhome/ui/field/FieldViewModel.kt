package com.gliksbot.axonhome.ui.field

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gliksbot.axonhome.controlplane.ControlPlaneClient
import com.gliksbot.axonhome.controlplane.Freshness
import com.gliksbot.axonhome.controlplane.LogicalRegion as CpRegion
import com.gliksbot.axonhome.controlplane.MaskKind
import com.gliksbot.axonhome.controlplane.MaskPolicy
import com.gliksbot.axonhome.controlplane.MaskState
import com.gliksbot.axonhome.controlplane.PutMasksRequest
import com.gliksbot.axonhome.core.domain.AxonJson
import com.gliksbot.axonhome.core.domain.CANONICAL_REGION_ORDER
import com.gliksbot.axonhome.core.domain.SharedFieldSnapshot
import com.gliksbot.axonhome.core.domain.wireName
import com.gliksbot.axonhome.core.sim.SIMULATION_MARKER
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.core.sim.TickRecord
import com.gliksbot.axonhome.sim.SimulationDriver
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.encodeToString

data class RegionUi(
    val name: String,
    val wireName: String,
    val charCount: Int,
    val spanCount: Int,
    val maskLabel: String,
    val spans: List<SpanUi>,
)

data class SpanUi(
    val spanId: String,
    val kind: String,
    val source: String,
    val provenance: String,
    val textPreview: String,
)

data class PipelineUi(
    val tickNumber: Int,
    val baseFieldId: String,
    val firstBoard: List<Pair<String, String>>,   // coreId → proposal text
    val refinedBoard: List<Pair<String, String>>,
    val consolidator: String,
    val verdictText: String?,
    val deltaIds: List<String>,
    val resultFieldId: String?,
    val rejected: Boolean,
    val rejectionReason: String?,
)

data class FieldState(
    val freshness: Freshness = Freshness.LIVE,
    val tickId: Int = 0,
    val fieldId: String = "",
    val canonicalHash: String = "",
    val regions: List<RegionUi> = emptyList(),
    val selectedRegion: Int = 0,
    val rawJson: String = "",
    // tick navigation over snapshots retained since app start
    val retainedTicks: Int = 0,
    val viewingTickOffset: Int = 0,   // 0 = head, 1 = previous tick, ...
    val pipeline: PipelineUi? = null,
    val inputDraft: String = "",
    val inputFeedback: String? = null,
)

class FieldViewModel(
    private val engine: SimulationEngine,
    private val client: ControlPlaneClient,
    private val driver: SimulationDriver,
) : ViewModel() {

    private val _state = MutableStateFlow(FieldState())
    val state: StateFlow<FieldState> = _state

    /** Snapshots retained since app start, head last. */
    private val retained = mutableListOf<SharedFieldSnapshot>()

    // ---- mask control state (two-step confirm, arch §10.F) ----
    data class MaskUiState(
        val current: MaskState? = null,
        val draft: Map<CpRegion, MaskPolicy> = emptyMap(),
        val proposalChanges: List<String> = emptyList(),
        val confirmToken: String? = null,
        val applied: Boolean = false,
        val busy: Boolean = false,
        val error: String? = null,
    )

    private val _maskState = MutableStateFlow(MaskUiState())
    val maskState: StateFlow<MaskUiState> = _maskState

    init {
        refresh()
        viewModelScope.launch { driver.tickCounter.collect { refresh() } }
    }

    fun refresh() {
        val head = engine.field
        if (retained.lastOrNull()?.fieldId != head.fieldId) retained += head
        val viewed = retained.getOrElse(retained.size - 1 - _state.value.viewingTickOffset) { head }
        val mask = engine.maskState
        _state.value = _state.value.copy(
            freshness = Freshness.LIVE,
            tickId = viewed.tickId,
            fieldId = viewed.fieldId,
            canonicalHash = viewed.canonicalHash,
            regions = viewed.regions.map { region ->
                RegionUi(
                    name = region.name.name,
                    wireName = region.name.wireName,
                    charCount = region.text.length,
                    spanCount = region.spans.size,
                    maskLabel = mask.policyFor(region.name).let { p ->
                        when (p.kind) {
                            "all" -> "all"
                            "none" -> "none"
                            "last_n_spans" -> "last ${p.limit} spans"
                            else -> "newest ${p.limit}%"
                        }
                    },
                    spans = region.spans.map { span ->
                        SpanUi(
                            spanId = span.spanId,
                            kind = span.kind,
                            source = span.source,
                            provenance = span.provenance,
                            textPreview = span.text.take(160),
                        )
                    },
                )
            },
            rawJson = AxonJson.encodeToString(viewed),
            retainedTicks = retained.size,
            pipeline = engine.tickHistory.lastOrNull()?.let { record -> toPipelineUi(record) },
        )
        viewModelScope.launch { refreshMasks() }
    }

    private fun toPipelineUi(record: TickRecord): PipelineUi = PipelineUi(
        tickNumber = record.tickNumber,
        baseFieldId = record.baseFieldId,
        firstBoard = record.firstProposals.map { it.authorCoreId to it.text },
        refinedBoard = record.refinedProposals.map { it.authorCoreId to it.text },
        consolidator = record.consolidatorCoreId,
        verdictText = record.verdict?.text,
        deltaIds = record.deltaIds,
        resultFieldId = record.resultFieldId,
        rejected = record.rejected,
        rejectionReason = record.rejectionReason,
    )

    fun selectRegion(index: Int) {
        _state.value = _state.value.copy(selectedRegion = index)
    }

    /** Navigate retained ticks: offset 0 = head, increasing = older. */
    fun viewTickOffset(offset: Int) {
        val clamped = offset.coerceIn(0, (retained.size - 1).coerceAtLeast(0))
        _state.value = _state.value.copy(viewingTickOffset = clamped)
        refreshKeepingOffset()
    }

    private fun refreshKeepingOffset() {
        val head = engine.field
        if (retained.lastOrNull()?.fieldId != head.fieldId) retained += head
        val viewed = retained.getOrElse(retained.size - 1 - _state.value.viewingTickOffset) { head }
        _state.value = _state.value.copy(
            tickId = viewed.tickId,
            fieldId = viewed.fieldId,
            canonicalHash = viewed.canonicalHash,
            retainedTicks = retained.size,
            rawJson = AxonJson.encodeToString(viewed),
        )
    }

    fun setInputDraft(text: String) {
        _state.value = _state.value.copy(inputDraft = text)
    }

    /** Durable ingress: submit user input into the sim field (audited by :core). */
    fun submitUserInput() {
        val text = _state.value.inputDraft
        if (text.isBlank()) return
        runCatching { engine.submitUserInput(text) }
            .onSuccess {
                _state.value = _state.value.copy(
                    inputDraft = "",
                    inputFeedback = "Committed to user_input via Heart valve delta [$SIMULATION_MARKER]",
                )
                driver.stepOnce()
                refresh()
            }
            .onFailure { error ->
                _state.value = _state.value.copy(inputFeedback = "Rejected: ${error.message}")
            }
    }

    // ---- mask control ----

    suspend fun refreshMasks() {
        val current = runCatching { client.masks() }.getOrNull() ?: return
        if (_maskState.value.draft.isEmpty()) {
            _maskState.value = _maskState.value.copy(current = current, draft = current.policies)
        } else {
            _maskState.value = _maskState.value.copy(current = current)
        }
    }

    /** Set a PROPOSED value only; nothing is applied until confirm. */
    fun draftMask(region: CpRegion, policy: MaskPolicy) {
        val next = _maskState.value.draft.toMutableMap()
        next[region] = policy
        _maskState.value = _maskState.value.copy(draft = next, applied = false, proposalChanges = emptyList(), confirmToken = null)
    }

    /** First PUT: obtain a proposal + confirm token; mutates nothing. */
    fun proposeMasks() {
        val draft = _maskState.value.draft
        if (draft.isEmpty()) return
        viewModelScope.launch {
            _maskState.value = _maskState.value.copy(busy = true, error = null)
            runCatching {
                client.putMasks(
                    PutMasksRequest(policies = draft, confirmed = false),
                    idempotencyKey = "mask-${System.currentTimeMillis()}",
                )
            }.onSuccess { response ->
                val changes = com.gliksbot.axonhome.ui.state.maskChanges(
                    _maskState.value.current ?: response.proposed,
                    response.proposed,
                )
                _maskState.value = _maskState.value.copy(
                    busy = false,
                    proposalChanges = changes,
                    confirmToken = response.confirmToken,
                )
            }.onFailure { error ->
                _maskState.value = _maskState.value.copy(busy = false, error = error.message)
            }
        }
    }

    /** Second PUT with the confirm token: applies the proposed policies. */
    fun confirmMasks() {
        val token = _maskState.value.confirmToken ?: return
        val draft = _maskState.value.draft
        viewModelScope.launch {
            _maskState.value = _maskState.value.copy(busy = true, error = null)
            runCatching {
                client.putMasks(
                    PutMasksRequest(policies = draft, confirmToken = token, confirmed = true),
                    idempotencyKey = "mask-confirm-${System.currentTimeMillis()}",
                )
            }.onSuccess { response ->
                _maskState.value = MaskUiState(
                    current = response.proposed,
                    draft = response.proposed.policies,
                    applied = true,
                )
                refresh()
            }.onFailure { error ->
                _maskState.value = _maskState.value.copy(busy = false, error = error.message)
            }
        }
    }

    fun revertMaskDraft() {
        val current = _maskState.value.current ?: return
        _maskState.value = MaskUiState(current = current, draft = current.policies)
    }
}
