package com.gliksbot.axonhome.core.domain

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Canonical Shared Field model — mirrors runtime/field/schema.py of
 * Axepapag/Axon @ main. Region names, order, schema string and canonical
 * hashing semantics are never renamed or reordered.
 */

const val SHARED_FIELD_SCHEMA = "shared-field-v4"

@Serializable
enum class LogicalRegion {
    @SerialName("conversation_history") CONVERSATION_HISTORY,
    @SerialName("user_input") USER_INPUT,
    @SerialName("cortex") CORTEX,
    @SerialName("situation_awareness") SITUATION_AWARENESS,
    @SerialName("tool_results") TOOL_RESULTS,
    @SerialName("advisor_input") ADVISOR_INPUT,
    @SerialName("task_state") TASK_STATE,
    @SerialName("scratch") SCRATCH,
    @SerialName("response_draft") RESPONSE_DRAFT,
    @SerialName("diary") DIARY,
    @SerialName("identity") IDENTITY,
    @SerialName("trainer_instructions") TRAINER_INSTRUCTIONS,
    @SerialName("training_responses") TRAINING_RESPONSES,
}

/** The 13 canonical regions in immutable canonical order (shared-field-v4). */
val CANONICAL_REGION_ORDER: List<LogicalRegion> = LogicalRegion.entries.toList()

/** Stable ordinal map, mirroring LOGICAL_REGION_IDS. */
val LOGICAL_REGION_IDS: Map<LogicalRegion, Int> =
    CANONICAL_REGION_ORDER.withIndex().associate { it.value to it.index }

val TRAINING_REGIONS: Set<LogicalRegion> =
    setOf(LogicalRegion.TRAINER_INSTRUCTIONS, LogicalRegion.TRAINING_RESPONSES)

/** Only these regions may be written by Core-authored deltas. */
val CORE_WRITABLE_REGIONS: Set<LogicalRegion> =
    setOf(LogicalRegion.SCRATCH, LogicalRegion.RESPONSE_DRAFT)

val LogicalRegion.wireName: String
    get() = AxonJson.encodeToString(LogicalRegion.serializer(), this).trim('"')

fun logicalRegionOf(value: String): LogicalRegion =
    CANONICAL_REGION_ORDER.firstOrNull { it.wireName == value }
        ?: throw IllegalArgumentException(
            "unknown logical region '$value'; dormant state is surfaced into " +
                "an active logical region before it can be attended"
        )

@Serializable
enum class RegionVisibility {
    @SerialName("attended") ATTENDED,
    @SerialName("masked") MASKED,
}

@Serializable
enum class WritePolicy {
    @SerialName("sealed") SEALED,
    @SerialName("core_writable") CORE_WRITABLE,
}

/** One contiguous attended character interval [start, end) inside a region. */
@Serializable
data class AttendedInterval(
    val start: Int,
    val end: Int,
) {
    init {
        require(start >= 0) { "AttendedInterval.start must be non-negative" }
        require(end >= start) { "AttendedInterval.end must be >= start" }
    }

    fun toCanonicalDict(): Map<String, Any?> = mapOf("start" to start, "end" to end)
}

/**
 * Reusable mask policy resolving one region to exact attended intervals.
 * kind in {all, none, last_n_spans, tail_percent}; tail_percent is the
 * operational 0-100 newest-suffix slider using integer ceiling arithmetic.
 */
@Serializable
data class RegionMaskPolicy(
    val kind: String,
    val limit: Int = 0,
) {
    init {
        require(kind.isNotEmpty()) { "RegionMaskPolicy.kind must be a non-empty string" }
        require(limit >= 0) { "RegionMaskPolicy.limit must be non-negative" }
        require(kind in MASK_KINDS) {
            "unsupported RegionMaskPolicy.kind '$kind'; expected one of $MASK_KINDS"
        }
        require(kind != KIND_TAIL_PERCENT || limit <= 100) {
            "tail_percent limit must be in [0, 100]"
        }
    }

    fun toCanonicalDict(): Map<String, Any?> = mapOf("kind" to kind, "limit" to limit)

    companion object {
        const val KIND_ALL = "all"
        const val KIND_NONE = "none"
        const val KIND_LAST_N_SPANS = "last_n_spans"
        const val KIND_TAIL_PERCENT = "tail_percent"
        val MASK_KINDS = setOf(KIND_ALL, KIND_NONE, KIND_LAST_N_SPANS, KIND_TAIL_PERCENT)

        fun all() = RegionMaskPolicy(KIND_ALL)
        fun none() = RegionMaskPolicy(KIND_NONE)
        fun lastNSpans(n: Int) = RegionMaskPolicy(KIND_LAST_N_SPANS, n)
        fun tailPercent(percent: Int) = RegionMaskPolicy(KIND_TAIL_PERCENT, percent)
    }
}

/**
 * Resolve a mask policy to explicit attended intervals over span text.
 * Pure function; mirrors schema.py::resolve_mask_policy exactly, including
 * the `(chars * percent + 99) / 100` ceiling for tail_percent.
 */
fun resolveMaskPolicy(spans: List<FieldSpan>, policy: RegionMaskPolicy): List<AttendedInterval> {
    if (policy.kind == RegionMaskPolicy.KIND_NONE) return emptyList()
    val textLength = spans.sumOf { it.text.codePointCount(0, it.text.length) }
    if (textLength == 0) return emptyList()
    return when (policy.kind) {
        RegionMaskPolicy.KIND_ALL -> listOf(AttendedInterval(0, textLength))
        RegionMaskPolicy.KIND_TAIL_PERCENT -> {
            if (policy.limit == 0) return emptyList()
            val attended = ((textLength.toLong() * policy.limit + 99) / 100).toInt()
            listOf(AttendedInterval(textLength - attended, textLength))
        }
        RegionMaskPolicy.KIND_LAST_N_SPANS -> {
            val total = spans.size
            if (policy.limit <= 0 || total == 0) return emptyList()
            val startSpan = maxOf(0, total - policy.limit)
            val intervals = mutableListOf<AttendedInterval>()
            var offset = 0
            spans.forEachIndexed { index, span ->
                if (index >= startSpan) intervals.add(AttendedInterval(offset, offset + span.text.codePointCount(0, span.text.length)))
                offset += span.text.codePointCount(0, span.text.length)
            }
            intervals
        }
        else -> throw IllegalArgumentException("unsupported RegionMaskPolicy.kind '${policy.kind}'")
    }
}

/** One immutable, provenance-bearing run of canonical characters. */
@Serializable
data class FieldSpan(
    @SerialName("span_id") val spanId: String,
    val text: String,
    val kind: String = "text",
    val source: String = "",
    val provenance: String = "",
    val confidence: Double = 1.0,
    @SerialName("container_refs") val containerRefs: List<String> = emptyList(),
    @SerialName("edge_refs") val edgeRefs: List<String> = emptyList(),
) {
    init {
        require(spanId.isNotEmpty()) { "FieldSpan.span_id must be a non-empty string" }
        require(text.isNotEmpty()) { "FieldSpan.text must be a non-empty string" }
        require(kind.isNotEmpty()) { "FieldSpan.kind must be a non-empty string" }
        require(confidence in 0.0..1.0) { "FieldSpan.confidence must be in [0, 1]" }
    }

    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "span_id" to spanId,
        "text" to text,
        "kind" to kind,
        "source" to source,
        "provenance" to provenance,
        "confidence" to confidence,
        "container_refs" to containerRefs.toSet().sorted(),
        "edge_refs" to edgeRefs.toSet().sorted(),
    )

    val canonicalHash: String get() = canonicalSha256(toCanonicalDict())
}

/**
 * Immutable state of one canonical logical region. The serialized
 * [attendedIntervals] / [maskPolicy] are the derived attention view; they are
 * excluded from the canonical hash (one-body doctrine: masks are views, not
 * canonical identity). A null [attendedIntervals] means "derive": from
 * [maskPolicy] when present, otherwise from [visibility].
 */
@Serializable
data class RegionState(
    val name: LogicalRegion,
    val spans: List<FieldSpan> = emptyList(),
    val visibility: RegionVisibility = RegionVisibility.ATTENDED,
    @SerialName("write_policy") val writePolicy: WritePolicy? = null,
    @SerialName("attended_intervals") val attendedIntervals: List<AttendedInterval>? = null,
    @SerialName("mask_policy") val maskPolicy: RegionMaskPolicy? = null,
) {
    init {
        require(spans.map { it.spanId }.toSet().size == spans.size) {
            "duplicate span_id in region '${name.wireName}'"
        }
        val policy = effectiveWritePolicy
        require(policy != WritePolicy.CORE_WRITABLE || name in CORE_WRITABLE_REGIONS) {
            "logical region '${name.wireName}' is always sealed"
        }
        val textLength = text.codePointCount(0, text.length)
        var previousEnd = 0
        attendedIntervals?.forEach { interval ->
            require(interval.start >= 0 && interval.end <= textLength) {
                "attended interval [${interval.start}, ${interval.end}) exceeds " +
                    "region '${name.wireName}' length $textLength"
            }
            require(interval.start >= previousEnd) {
                "attended intervals in region '${name.wireName}' must be sorted and non-overlapping"
            }
            previousEnd = interval.end
        }
    }

    val effectiveWritePolicy: WritePolicy
        get() = writePolicy
            ?: if (name in CORE_WRITABLE_REGIONS) WritePolicy.CORE_WRITABLE else WritePolicy.SEALED

    /** The complete canonical string; no view/window truncation applies. */
    val text: String get() = spans.joinToString("") { it.text }

    /** Exact attended intervals (stored, or derived from policy/visibility). */
    val resolvedAttendedIntervals: List<AttendedInterval>
        get() = attendedIntervals ?: maskPolicy?.let { resolveMaskPolicy(spans, it) }
            ?: if (visibility == RegionVisibility.ATTENDED) {
                if (text.isEmpty()) emptyList() else listOf(AttendedInterval(0, text.codePointCount(0, text.length)))
            } else {
                emptyList()
            }

    val attendedText: String
        get() {
            val full = text
            return resolvedAttendedIntervals.joinToString("") { full.substring(full.offsetByCodePoints(0, it.start), full.offsetByCodePoints(0, it.end)) }
        }

    fun withSpans(newSpans: List<FieldSpan>): RegionState = copy(
        spans = newSpans,
        attendedIntervals = null, // re-resolve from mask policy / visibility
    )

    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "name" to name.wireName,
        "visibility" to AxonJson.encodeToString(RegionVisibility.serializer(), visibility).trim('"'),
        "write_policy" to AxonJson.encodeToString(WritePolicy.serializer(), effectiveWritePolicy).trim('"'),
        "spans" to spans.map { it.toCanonicalDict() },
    )

    val canonicalHash: String get() = canonicalSha256(toCanonicalDict())

    companion object {
        fun fromText(
            name: LogicalRegion,
            text: String,
            spanId: String = "${name.wireName}-span-0",
            kind: String = "text",
            source: String = "",
            provenance: String = "",
        ): RegionState = RegionState(
            name = name,
            spans = if (text.isEmpty()) emptyList() else listOf(
                FieldSpan(spanId = spanId, text = text, kind = kind, source = source, provenance = provenance)
            ),
        )
    }
}

/**
 * A complete immutable field state linked to its parent by hash.
 * [fieldId] = canonicalSha256 of the canonical dict (tick id, ordered
 * regions, parent id, source manifest ids). Masks never enter the hash.
 */
@Serializable
data class SharedFieldSnapshot(
    @SerialName("tick_id") val tickId: Int,
    val regions: List<RegionState> = emptyList(),
    @SerialName("parent_field_id") val parentFieldId: String? = null,
    @SerialName("source_manifest_ids") val sourceManifestIds: List<String> = emptyList(),
    @SerialName("schema") val schema: String = SHARED_FIELD_SCHEMA,
) {
    init {
        require(schema == SHARED_FIELD_SCHEMA) {
            "unsupported shared-field schema '$schema'; expected '$SHARED_FIELD_SCHEMA'"
        }
        require(tickId >= 0) { "SharedFieldSnapshot.tick_id must be non-negative" }
        require(parentFieldId == null || parentFieldId.isNotEmpty()) {
            "parent_field_id must be null or a non-empty string"
        }
        require(sourceManifestIds.none { it.isEmpty() }) {
            "source_manifest_ids cannot contain empty values"
        }
        val names = regions.map { it.name }
        require(names.toSet().size == names.size) { "duplicate logical region in snapshot" }
        require(names == CANONICAL_REGION_ORDER) {
            "snapshot regions must be exactly the 13 canonical regions in canonical order"
        }
    }

    fun region(name: LogicalRegion): RegionState = regions[LOGICAL_REGION_IDS.getValue(name)]

    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to schema,
        "tick_id" to tickId,
        "parent_field_id" to parentFieldId,
        "source_manifest_ids" to sourceManifestIds.toSet().sorted(),
        "regions" to regions.map { it.toCanonicalDict() },
    )

    val fieldId: String get() = canonicalSha256(toCanonicalDict())
    val canonicalHash: String get() = fieldId

    companion object {
        fun create(
            tickId: Int = 0,
            regions: Map<LogicalRegion, RegionState> = emptyMap(),
            parentFieldId: String? = null,
            sourceManifestIds: List<String> = emptyList(),
        ): SharedFieldSnapshot = SharedFieldSnapshot(
            tickId = tickId,
            regions = CANONICAL_REGION_ORDER.map { regions[it] ?: RegionState(name = it) },
            parentFieldId = parentFieldId,
            sourceManifestIds = sourceManifestIds.toSet().sorted(),
        )

        fun empty(tickId: Int = 0, parentFieldId: String? = null): SharedFieldSnapshot =
            create(tickId = tickId, parentFieldId = parentFieldId)

        fun fromTexts(
            texts: Map<LogicalRegion, String>,
            tickId: Int = 0,
            parentFieldId: String? = null,
            source: String = "",
            provenance: String = "",
        ): SharedFieldSnapshot = create(
            tickId = tickId,
            regions = texts.mapValues { (region, text) ->
                RegionState.fromText(region, text, source = source, provenance = provenance)
            },
            parentFieldId = parentFieldId,
        )
    }
}
