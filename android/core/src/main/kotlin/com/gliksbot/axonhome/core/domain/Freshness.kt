package com.gliksbot.axonhome.core.domain

import kotlinx.serialization.Serializable

/**
 * Freshness labeling — every screen-facing payload carries an explicit
 * freshness state. Cached state must never be presented as current.
 */
@Serializable
enum class StateFreshness {
    LIVE,
    CACHED,
    STALE,
    DISCONNECTED,
}

/**
 * Wrapper for every payload crossing to the UI: [data], its [freshness]
 * label, the [asOf] timestamp the data describes, and the [source] that
 * produced it (e.g. "heart-host", "mirror-cache", "SIMULATION").
 */
@Serializable
data class MirrorEnvelope<T>(
    val data: T,
    val freshness: StateFreshness,
    val asOf: String,
    val source: String,
) {
    fun <R> map(transform: (T) -> R): MirrorEnvelope<R> =
        MirrorEnvelope(transform(data), freshness, asOf, source)
}
