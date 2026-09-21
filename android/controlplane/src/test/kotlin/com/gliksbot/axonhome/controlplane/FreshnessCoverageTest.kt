package com.gliksbot.axonhome.controlplane

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Freshness enum coverage (arch §4.3/§12): the wire contract has exactly four
 * freshness states; SIMULATION is a UI-side state and must NOT appear in this
 * API.
 */
class FreshnessCoverageTest {

    private val json = Json { ignoreUnknownKeys = true }

    @Test
    fun `freshness has exactly the four wire states`() {
        assertEquals(
            setOf(Freshness.LIVE, Freshness.CACHED, Freshness.STALE, Freshness.DISCONNECTED),
            Freshness.entries.toSet(),
        )
        // SIMULATION is deliberately absent from the API contract.
        assertTrue(Freshness.entries.none { it.name == "SIMULATION" })
    }

    @Test
    fun `every freshness value survives wire encoding`() {
        Freshness.entries.forEach { freshness ->
            val encoded = json.encodeToString(freshness)
            assertEquals("\"${freshness.name}\"", encoded)
            assertEquals(freshness, json.decodeFromString<Freshness>(encoded))
        }
    }

    @Test
    fun `freshness schema enum matches the Kotlin enum`() {
        // The JSON Schema (draft 2020-12) enum in common.json must list exactly
        // the Kotlin Freshness wire values — this is the contract<->model pin.
        val common = SchemaCatalog.loadJson("common")
        val schemaValues = common.getValue("\$defs").jsonObject
            .getValue("freshness").jsonObject
            .getValue("enum").jsonArray
            .map { it.jsonPrimitive.content }
            .toSet()
        assertEquals(Freshness.entries.map { it.name }.toSet(), schemaValues)
    }
}
