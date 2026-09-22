@file:OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)

package com.gliksbot.axonhome.core.domain

import kotlinx.serialization.Serializable
import kotlinx.serialization.SerialName
import kotlinx.serialization.json.JsonClassDiscriminator

/**
 * FieldDelta — mirrors runtime/field/delta.py of Axepapag/Axon @ main.
 * A delta is authored against exactly one immutable base snapshot; Heart is
 * the only commit path. Validation rejects stale bases, sealed-region
 * writes, out-of-bounds operations and overlapping operations.
 */

const val FIELD_DELTA_SCHEMA = "shared-field-delta-v1"

open class DeltaValidationError(message: String) : IllegalArgumentException(message)
class StaleDeltaError(message: String) : DeltaValidationError(message)
class SealedRegionWriteError(message: String) : DeltaValidationError(message)
class OverlappingDeltaError(message: String) : DeltaValidationError(message)

@Serializable
@JsonClassDiscriminator("op")
sealed class FieldOperation {
    abstract val region: LogicalRegion
    abstract val provenance: String
    abstract val containerRefs: List<String>
    abstract val edgeRefs: List<String>
    abstract val start: Int
    abstract val end: Int
    abstract val replacementText: String
    abstract val opName: String

    abstract fun toCanonicalDict(): Map<String, Any?>
}

@Serializable
@SerialName("insert")
data class InsertText(
    override val region: LogicalRegion,
    val offset: Int,
    val text: String,
    override val provenance: String = "",
    @SerialName("container_refs") override val containerRefs: List<String> = emptyList(),
    @SerialName("edge_refs") override val edgeRefs: List<String> = emptyList(),
) : FieldOperation() {
    init {
        require(offset >= 0) { "InsertText.offset must be non-negative" }
        require(text.isNotEmpty()) { "InsertText.text must be non-empty" }
    }

    override val start: Int get() = offset
    override val end: Int get() = offset
    override val replacementText: String get() = text
    override val opName: String get() = "insert"

    override fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "op" to "insert",
        "region" to region.wireName,
        "offset" to offset,
        "text" to text,
        "provenance" to provenance,
        "container_refs" to containerRefs.toSet().sorted(),
        "edge_refs" to edgeRefs.toSet().sorted(),
    )
}

@Serializable
@SerialName("delete")
data class DeleteText(
    override val region: LogicalRegion,
    @SerialName("start") val startBound: Int,
    @SerialName("end") val endBound: Int,
    override val provenance: String = "",
    @SerialName("container_refs") override val containerRefs: List<String> = emptyList(),
    @SerialName("edge_refs") override val edgeRefs: List<String> = emptyList(),
) : FieldOperation() {
    init {
        require(startBound >= 0 && endBound > startBound) { "DeleteText requires 0 <= start < end" }
    }

    override val start: Int get() = startBound
    override val end: Int get() = endBound
    override val replacementText: String get() = ""
    override val opName: String get() = "delete"

    override fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "op" to "delete",
        "region" to region.wireName,
        "start" to startBound,
        "end" to endBound,
        "provenance" to provenance,
        "container_refs" to containerRefs.toSet().sorted(),
        "edge_refs" to edgeRefs.toSet().sorted(),
    )
}

@Serializable
@SerialName("replace")
data class ReplaceText(
    override val region: LogicalRegion,
    @SerialName("start") val startBound: Int,
    @SerialName("end") val endBound: Int,
    val text: String,
    override val provenance: String = "",
    @SerialName("container_refs") override val containerRefs: List<String> = emptyList(),
    @SerialName("edge_refs") override val edgeRefs: List<String> = emptyList(),
) : FieldOperation() {
    init {
        require(startBound >= 0 && endBound >= startBound) { "ReplaceText requires 0 <= start <= end" }
        require(!(startBound == endBound && text.isEmpty())) { "ReplaceText cannot be an empty no-op" }
    }

    override val start: Int get() = startBound
    override val end: Int get() = endBound
    override val replacementText: String get() = text
    override val opName: String get() = "replace"

    override fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "op" to "replace",
        "region" to region.wireName,
        "start" to startBound,
        "end" to endBound,
        "text" to text,
        "provenance" to provenance,
        "container_refs" to containerRefs.toSet().sorted(),
        "edge_refs" to edgeRefs.toSet().sorted(),
    )
}

private fun isInsertLike(op: FieldOperation): Boolean = op.start == op.end

private fun operationsOverlap(left: FieldOperation, right: FieldOperation): Boolean {
    val leftInsert = isInsertLike(left)
    val rightInsert = isInsertLike(right)
    if (leftInsert && rightInsert) return left.start == right.start
    if (leftInsert) return right.start < left.start && left.start < right.end
    if (rightInsert) return left.start < right.start && right.start < left.end
    return maxOf(left.start, right.start) < minOf(left.end, right.end)
}

/** A proposal authored against exactly one immutable base snapshot. */
@Serializable
data class FieldDelta(
    @SerialName("base_field_id") val baseFieldId: String,
    @SerialName("base_tick_id") val baseTickId: Int,
    @SerialName("author_core_id") val authorCoreId: String,
    @SerialName("pass_id") val passId: String,
    val operations: List<FieldOperation>,
    val evidence: List<String> = emptyList(),
    @SerialName("schema") val schema: String = FIELD_DELTA_SCHEMA,
) {
    init {
        require(baseFieldId.isNotEmpty()) { "FieldDelta.base_field_id must be non-empty" }
        require(baseTickId >= 0) { "FieldDelta.base_tick_id must be non-negative" }
        require(authorCoreId.isNotEmpty()) { "FieldDelta.author_core_id must be non-empty" }
        require(passId.isNotEmpty()) { "FieldDelta.pass_id must be non-empty" }
        require(operations.isNotEmpty()) { "FieldDelta.operations must be non-empty" }
        require(schema == FIELD_DELTA_SCHEMA) { "unsupported delta schema '$schema'" }
    }

    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to FIELD_DELTA_SCHEMA,
        "base_field_id" to baseFieldId,
        "base_tick_id" to baseTickId,
        "author_core_id" to authorCoreId,
        "pass_id" to passId,
        "operations" to operations.map { it.toCanonicalDict() },
        "evidence" to evidence.toSet().sorted(),
    )

    val deltaId: String get() = canonicalSha256(toCanonicalDict())
    val canonicalHash: String get() = deltaId
}

/**
 * Validate [delta] against [snapshot]: staleness, sealed-region authority,
 * bounds, and per-region overlap. [permittedRegions] widens the writable set
 * beyond CORE_WRITABLE_REGIONS (e.g. a Heart authority grant).
 */
fun validateDelta(
    snapshot: SharedFieldSnapshot,
    delta: FieldDelta,
    permittedRegions: Set<LogicalRegion>? = null,
) {
    if (delta.baseFieldId != snapshot.fieldId || delta.baseTickId != snapshot.tickId) {
        throw StaleDeltaError("delta base does not match the current field id and tick")
    }
    val allowed = CORE_WRITABLE_REGIONS + (permittedRegions ?: emptySet())

    val byRegion = LinkedHashMap<LogicalRegion, MutableList<FieldOperation>>()
    for (operation in delta.operations) {
        val regionName = operation.region
        if (regionName !in allowed) {
            throw SealedRegionWriteError("logical region '${regionName.wireName}' is sealed")
        }
        val region = snapshot.region(regionName)
        if (operation.start < 0 || operation.end < operation.start || operation.end > region.text.length) {
            throw DeltaValidationError(
                "operation bounds [${operation.start}, ${operation.end}) exceed " +
                    "'${regionName.wireName}' length ${region.text.length}"
            )
        }
        byRegion.getOrPut(regionName) { mutableListOf() }.add(operation)
    }

    for ((regionName, operations) in byRegion) {
        for (i in operations.indices) {
            for (j in i + 1 until operations.size) {
                if (operationsOverlap(operations[i], operations[j])) {
                    throw OverlappingDeltaError("overlapping operations in '${regionName.wireName}'")
                }
            }
        }
    }
}

private fun sliceSpans(region: RegionState, start: Int, end: Int): List<FieldSpan> {
    if (start == end) return emptyList()
    val result = mutableListOf<FieldSpan>()
    var regionOffset = 0
    for (span in region.spans) {
        val spanStart = regionOffset
        val spanEnd = spanStart + span.text.length
        val iStart = maxOf(start, spanStart)
        val iEnd = minOf(end, spanEnd)
        if (iStart < iEnd) {
            val localStart = iStart - spanStart
            val localEnd = iEnd - spanStart
            if (localStart == 0 && localEnd == span.text.length) {
                result.add(span)
            } else {
                result.add(
                    span.copy(
                        spanId = "${span.spanId}@$localStart:$localEnd",
                        text = span.text.substring(localStart, localEnd),
                    )
                )
            }
        }
        regionOffset = spanEnd
        if (regionOffset >= end) break
    }
    return result
}

private fun applyRegionOperations(
    region: RegionState,
    indexedOperations: List<Pair<Int, FieldOperation>>,
    delta: FieldDelta,
): RegionState {
    val ordered = indexedOperations.sortedWith(
        compareBy({ it.second.start }, { if (isInsertLike(it.second)) 0 else 1 }, { it.first })
    )
    val spans = mutableListOf<FieldSpan>()
    var cursor = 0
    for ((operationIndex, operation) in ordered) {
        spans.addAll(sliceSpans(region, cursor, operation.start))
        val replacement = operation.replacementText
        if (replacement.isNotEmpty()) {
            spans.add(
                FieldSpan(
                    spanId = "delta-${delta.deltaId.take(20)}-${"%04d".format(operationIndex)}",
                    text = replacement,
                    kind = "delta_${operation.opName}",
                    source = delta.authorCoreId,
                    provenance = operation.provenance.ifEmpty {
                        "field_delta:${delta.deltaId}:op:$operationIndex"
                    },
                    containerRefs = operation.containerRefs,
                    edgeRefs = operation.edgeRefs,
                )
            )
        }
        // Insertions consume no source characters; the untouched prefix before
        // their boundary has already been emitted.
        cursor = operation.end
    }
    spans.addAll(sliceSpans(region, cursor, region.text.length))
    return region.withSpans(spans)
}

/** Validate and apply [delta] as one deterministic transaction. */
fun applyDelta(
    snapshot: SharedFieldSnapshot,
    delta: FieldDelta,
    permittedRegions: Set<LogicalRegion>? = null,
): SharedFieldSnapshot {
    validateDelta(snapshot, delta, permittedRegions)
    val indexedByRegion = LinkedHashMap<LogicalRegion, MutableList<Pair<Int, FieldOperation>>>()
    delta.operations.forEachIndexed { index, operation ->
        indexedByRegion.getOrPut(operation.region) { mutableListOf() }.add(index to operation)
    }
    val nextRegions = snapshot.regions.map { region ->
        val ops = indexedByRegion[region.name]
        if (ops == null) region else applyRegionOperations(region, ops, delta)
    }
    return SharedFieldSnapshot(
        tickId = snapshot.tickId + 1,
        regions = nextRegions,
        parentFieldId = snapshot.fieldId,
        sourceManifestIds = snapshot.sourceManifestIds,
    )
}

fun replayDeltas(base: SharedFieldSnapshot, deltas: List<FieldDelta>): SharedFieldSnapshot {
    var snapshot = base
    for (delta in deltas) snapshot = applyDelta(snapshot, delta)
    return snapshot
}
