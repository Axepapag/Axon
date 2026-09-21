package com.gliksbot.axonhome.controlplane

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject

/**
 * Loads the language-neutral JSON Schema files (JSON Schema draft 2020-12)
 * bundled as classpath resources under `schemas/`. The resources are
 * byte-identical copies of `controlplane/schemas/` at the repository root;
 * `SchemaResourcesConsistencyTest` enforces that in the source tree.
 *
 * This catalog returns raw schema documents; validation itself is left to the
 * caller (or a JSON-Schema engine chosen by :app) so this module stays
 * dependency-free beyond kotlinx-serialization.
 */
object SchemaCatalog {

    const val SCHEMA_ID_PREFIX = "https://axon.gliksbot.com/schemas/controlplane/v1/"
    private const val RESOURCE_ROOT = "/schemas"

    /** The 13 message-group schema files (one per group). */
    val GROUPS: List<String> = listOf(
        "common",
        "field",
        "masks",
        "cores",
        "souls",
        "trainer",
        "compute",
        "storage",
        "capsules",
        "audit",
        "agents",
        "events",
        "stop",
    )

    /** Raw schema document text for a group, e.g. `load("events")`. */
    fun load(group: String): String {
        require(group in GROUPS) { "unknown schema group '$group'; known: $GROUPS" }
        val path = "$RESOURCE_ROOT/$group.json"
        return checkNotNull(SchemaCatalog::class.java.getResourceAsStream(path)) {
            "schema resource missing from classpath: $path"
        }.bufferedReader().use { it.readText() }
    }

    /** Parsed schema document. */
    fun loadJson(group: String, json: Json = Json): JsonObject =
        json.parseToJsonElement(load(group)) as JsonObject

    /** The `$id` declared by a group's schema, e.g. https://.../controlplane/v1/events.json. */
    fun schemaId(group: String): String = "$SCHEMA_ID_PREFIX$group.json"

    /** All groups, loaded and keyed by group name. */
    fun loadAll(): Map<String, JsonObject> = GROUPS.associateWith { loadJson(it) }
}
