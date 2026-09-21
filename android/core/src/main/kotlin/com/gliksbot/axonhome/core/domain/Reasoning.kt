package com.gliksbot.axonhome.core.domain

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * English reasoning contracts — mirrors runtime/heart/english_reasoning.py.
 * FIRST and REFINED are ordinary language; FINAL is tagged desired-state
 * language that Heart alone materializes into a typed FieldDelta.
 */

const val ENGLISH_PROPOSAL_SCHEMA = "axon-english-proposal-v1"
const val TECHNICAL_FINAL_VERDICT_SCHEMA = "axon-tagged-final-verdict-v2"

class EnglishReasoningContractError(message: String) : IllegalArgumentException(message)

/** Canonical FINAL tag surface, in canonical region order. */
val REGION_TAGS: Map<LogicalRegion, String> = linkedMapOf(
    LogicalRegion.CONVERSATION_HISTORY to "conversationHistory",
    LogicalRegion.USER_INPUT to "userInput",
    LogicalRegion.CORTEX to "cortex",
    LogicalRegion.SITUATION_AWARENESS to "situationAwareness",
    LogicalRegion.TOOL_RESULTS to "toolResults",
    LogicalRegion.ADVISOR_INPUT to "advisorInput",
    LogicalRegion.TASK_STATE to "taskState",
    LogicalRegion.SCRATCH to "scratch",
    LogicalRegion.RESPONSE_DRAFT to "responseDraft",
    LogicalRegion.DIARY to "journal",
    LogicalRegion.IDENTITY to "identity",
    LogicalRegion.TRAINER_INSTRUCTIONS to "trainerInstructions",
    LogicalRegion.TRAINING_RESPONSES to "trainingResponses",
)

val TAG_TO_REGION: Map<String, LogicalRegion> =
    REGION_TAGS.entries.associate { (region, tag) -> tag to region } +
        mapOf("diary" to LogicalRegion.DIARY) // accepted compatibility alias

private val TAG_LINE = Regex("^#([A-Za-z][A-Za-z0-9]*)#(?: (.*))?$")

private fun looksLikeTagLine(line: String): Boolean = TAG_LINE.matches(line)

private fun escapeBodyLine(line: String): String =
    // A leading backslash escapes itself; a line that looks like a region tag
    // is escaped so it cannot be parsed as a section header.
    if (line.startsWith("\\") || looksLikeTagLine(line)) "\\$line" else line

private fun unescapeBodyLine(line: String): String =
    if (line.startsWith("\\")) line.substring(1) else line

private fun requireExactText(value: String, name: String, nonempty: Boolean): String {
    if (nonempty && value.isBlank()) {
        throw EnglishReasoningContractError("$name must be nonempty English text")
    }
    return value
}

fun coerceRegionFromTag(value: String): LogicalRegion =
    TAG_TO_REGION[value] ?: try {
        logicalRegionOf(value)
    } catch (e: IllegalArgumentException) {
        throw EnglishReasoningContractError("unknown final-verdict region '$value'")
    }

/** Render one region section as `#tag# first-line` plus escaped body lines. */
fun renderRegionSection(region: LogicalRegion, text: String): List<String> {
    val tag = REGION_TAGS.getValue(region)
    val bodyLines = text.split("\n")
    val first = bodyLines.firstOrNull().orEmpty()
    val lines = mutableListOf("#$tag#" + if (first.isNotEmpty()) " $first" else "")
    bodyLines.drop(1).forEach { lines.add(escapeBodyLine(it)) }
    return lines
}

/** Render a tagged FINAL verdict from ordered region sections. */
fun renderFinalVerdict(sections: List<Pair<LogicalRegion, String>>): String {
    require(sections.isNotEmpty()) { "final verdict must contain at least one #region# section" }
    return sections.flatMap { (region, text) -> renderRegionSection(region, text) }
        .joinToString("\n")
}

/** Parse a tagged FINAL verdict into ordered (region, text) sections. */
fun parseFinalVerdictSections(text: String): List<Pair<LogicalRegion, String>> {
    requireExactText(text, "final verdict", nonempty = true)
    val sections = mutableListOf<Pair<LogicalRegion, String>>()
    val seen = mutableSetOf<LogicalRegion>()
    var currentRegion: LogicalRegion? = null
    val currentLines = mutableListOf<String>()

    fun flush() {
        val region = currentRegion ?: return
        sections.add(region to currentLines.joinToString("\n"))
        currentRegion = null
        currentLines.clear()
    }

    text.split("\n").forEachIndexed { index, line ->
        val match = TAG_LINE.matchEntire(line)
        val tagRegion = match?.groupValues?.get(1)?.let { tag ->
            TAG_TO_REGION[tag] ?: throw EnglishReasoningContractError(
                "unknown final-verdict region tag '#$tag#'"
            )
        }
        if (match != null && tagRegion != null) {
            flush()
            if (!seen.add(tagRegion)) {
                throw EnglishReasoningContractError(
                    "duplicate final-verdict section for region '${tagRegion.wireName}'"
                )
            }
            currentRegion = tagRegion
            val firstLine = match.groupValues[2]
            currentLines.add(unescapeBodyLine(firstLine))
        } else {
            if (currentRegion == null) {
                throw EnglishReasoningContractError(
                    "final verdict line ${index + 1} appears before any #region# tag"
                )
            }
            currentLines.add(unescapeBodyLine(line))
        }
    }
    flush()
    if (sections.isEmpty()) {
        throw EnglishReasoningContractError("final verdict must contain at least one #region# section")
    }
    return sections
}

/** One exact nonempty FIRST or REFINED proposal authored by a reasoning core. */
@Serializable
data class EnglishProposal(
    @SerialName("base_field_id") val baseFieldId: String,
    @SerialName("base_tick_id") val baseTickId: Int,
    @SerialName("author_core_id") val authorCoreId: String,
    @SerialName("pass_id") val passId: String,
    @SerialName("rail_d_model") val railDModel: Int,
    val text: String,
    val evidence: List<String> = emptyList(),
    @SerialName("schema") val schema: String = ENGLISH_PROPOSAL_SCHEMA,
) {
    init {
        if (baseFieldId.isEmpty()) throw EnglishReasoningContractError("base_field_id must be nonempty")
        if (authorCoreId.isEmpty()) throw EnglishReasoningContractError("author_core_id must be nonempty")
        if (baseTickId < 0) throw EnglishReasoningContractError("base_tick_id must be a non-negative integer")
        if (passId !in setOf("first", "refined")) {
            throw EnglishReasoningContractError("EnglishProposal.pass_id must be 'first' or 'refined'")
        }
        if (railDModel < 16 || railDModel % 16 != 0) {
            throw EnglishReasoningContractError("rail_d_model must be a positive multiple of 16")
        }
        if (schema != ENGLISH_PROPOSAL_SCHEMA) {
            throw EnglishReasoningContractError("serialized English proposal schema is invalid")
        }
        requireExactText(text, "proposal text", nonempty = true)
    }

    fun toCanonicalDict(includeId: Boolean = true): Map<String, Any?> {
        val value = mutableMapOf<String, Any?>(
            "schema" to ENGLISH_PROPOSAL_SCHEMA,
            "base_field_id" to baseFieldId,
            "base_tick_id" to baseTickId,
            "author_core_id" to authorCoreId,
            "pass_id" to passId,
            "rail_d_model" to railDModel,
            "text" to text,
            "evidence" to evidence.toSet().sorted(),
        )
        if (includeId) value["proposal_id"] = proposalId
        return value
    }

    val proposalId: String get() = canonicalSha256(toCanonicalDict(includeId = false))
}

/**
 * One tagged-region FINAL utterance authored by the rotating consolidator.
 * The core emits only [text]; binding metadata is supplied by the runtime
 * envelope. Heart materializes [text] against the frozen base into an
 * internal FieldDelta only after receipt.
 */
@Serializable
data class TechnicalFinalVerdict(
    @SerialName("base_field_id") val baseFieldId: String,
    @SerialName("base_tick_id") val baseTickId: Int,
    @SerialName("author_core_id") val authorCoreId: String,
    @SerialName("rail_d_model") val railDModel: Int,
    val text: String,
    val evidence: List<String> = emptyList(),
    @SerialName("schema") val schema: String = TECHNICAL_FINAL_VERDICT_SCHEMA,
) {
    init {
        if (baseFieldId.isEmpty()) throw EnglishReasoningContractError("base_field_id must be nonempty")
        if (authorCoreId.isEmpty()) throw EnglishReasoningContractError("author_core_id must be nonempty")
        if (baseTickId < 0) throw EnglishReasoningContractError("base_tick_id must be a non-negative integer")
        if (railDModel < 16 || railDModel % 16 != 0) {
            throw EnglishReasoningContractError("rail_d_model must be a positive multiple of 16")
        }
        if (schema != TECHNICAL_FINAL_VERDICT_SCHEMA) {
            throw EnglishReasoningContractError("serialized FINAL verdict schema is invalid")
        }
        // Syntax is checked without needing semantic state; actual change and
        // authority are checked when Heart binds the frozen base.
        parseFinalVerdictSections(text)
    }

    fun toCanonicalDict(includeId: Boolean = true): Map<String, Any?> {
        val value = mutableMapOf<String, Any?>(
            "schema" to TECHNICAL_FINAL_VERDICT_SCHEMA,
            "base_field_id" to baseFieldId,
            "base_tick_id" to baseTickId,
            "author_core_id" to authorCoreId,
            "rail_d_model" to railDModel,
            "text" to text,
            "evidence" to evidence.toSet().sorted(),
        )
        if (includeId) value["verdict_id"] = verdictId
        return value
    }

    val verdictId: String get() = canonicalSha256(toCanonicalDict(includeId = false))

    /**
     * Whole-region materialization: every tagged section becomes a
     * ReplaceText over the full region body (InsertText at 0 for an empty
     * region); unchanged regions emit no operation.
     */
    fun materialize(base: SharedFieldSnapshot): FieldDelta {
        if (base.fieldId != baseFieldId || base.tickId != baseTickId) {
            throw EnglishReasoningContractError("final verdict is stale for the supplied frozen base")
        }
        val operations = mutableListOf<FieldOperation>()
        for ((region, text) in parseFinalVerdictSections(text)) {
            val existing = base.region(region).text
            if (existing == text) continue
            operations.add(
                if (existing.isEmpty()) {
                    InsertText(region = region, offset = 0, text = text, provenance = "final_verdict:$verdictId")
                } else {
                    ReplaceText(
                        region = region,
                        startBound = 0,
                        endBound = existing.length,
                        text = text,
                        provenance = "final_verdict:$verdictId",
                    )
                }
            )
        }
        if (operations.isEmpty()) {
            throw EnglishReasoningContractError("final verdict proposes no change to the frozen base")
        }
        return FieldDelta(
            baseFieldId = baseFieldId,
            baseTickId = baseTickId,
            authorCoreId = authorCoreId,
            passId = "consolidated",
            operations = operations,
            evidence = evidence,
        )
    }
}
