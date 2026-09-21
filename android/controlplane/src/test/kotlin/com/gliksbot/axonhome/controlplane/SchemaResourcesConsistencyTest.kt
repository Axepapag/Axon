package com.gliksbot.axonhome.controlplane

import java.security.MessageDigest
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Pins the classpath schema resources to the repo-level language-neutral
 * schemas: `src/main/resources/schemas/<group>.json` must be byte-identical to
 * `controlplane/schemas/<group>.json`. If you edit one side, copy it to the
 * other — this test fails otherwise.
 */
class SchemaResourcesConsistencyTest {

    @Test
    fun `every schema group resource exists and declares the expected $id`() {
        for (group in SchemaCatalog.GROUPS) {
            val doc = SchemaCatalog.loadJson(group)
            assertEquals(
                SchemaCatalog.schemaId(group),
                doc["\$id"]!!.jsonPrimitive.content,
                "schema $group has wrong \$id",
            )
        }
    }

    @Test
    fun `classpath resources are byte-identical to the repo-level schemas`() {
        val repoSchemas = ControlPlanePaths.schemasDir()
        assertTrue(repoSchemas.isDirectory, "repo schema dir missing: $repoSchemas")
        for (group in SchemaCatalog.GROUPS) {
            val repoFile = java.io.File(repoSchemas, "$group.json")
            assertTrue(repoFile.isFile, "missing repo schema: ${repoFile.absolutePath}")
            val repoBytes = repoFile.readBytes()
            val resourceBytes = SchemaCatalog.load(group).toByteArray(Charsets.UTF_8)
            assertEquals(
                sha256(repoBytes), sha256(resourceBytes),
                "schemas/$group.json differs between controlplane/schemas/ and classpath resources — copy the file",
            )
        }
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it) }
}
