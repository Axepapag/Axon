package com.gliksbot.axonhome.core.capsule

import com.gliksbot.axonhome.core.domain.AcceptedStepBundle
import com.gliksbot.axonhome.core.domain.AnatomyConfig
import com.gliksbot.axonhome.core.domain.AxonJson
import com.gliksbot.axonhome.core.domain.CanonicalJson
import com.gliksbot.axonhome.core.domain.CheckpointRecord
import com.gliksbot.axonhome.core.domain.LearningPolicy
import com.gliksbot.axonhome.core.domain.ResourceTranche
import com.gliksbot.axonhome.core.domain.SoulSnapshot
import com.gliksbot.axonhome.core.domain.canonicalSha256
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString

/**
 * Recovery capsule (mission §15/§O) — a self-verifying, content-addressed
 * bundle carrying the minimum complete state required to continue the
 * organism. Mirrors the axon-cloud-bundle-manifest-v1 philosophy:
 * deterministic members, a detached manifest written LAST, and fail-closed
 * verification — nothing half-verified reaches canonical outputs.
 */

const val CAPSULE_MANIFEST_SCHEMA = "axon-home-recovery-capsule-manifest-v1"
const val MANIFEST_MEMBER_NAME = "manifest.json"

@Serializable
data class CapsuleMember(
    val sha256: String,
    val bytes: Long,
)

@Serializable
data class CapsuleManifest(
    @SerialName("capsule_id") val capsuleId: String,
    val kind: String,
    @SerialName("created_at") val createdAt: String,
    val members: Map<String, CapsuleMember>,
    val details: Map<String, String> = emptyMap(),
    @SerialName("schema") val schema: String = CAPSULE_MANIFEST_SCHEMA,
) {
    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to CAPSULE_MANIFEST_SCHEMA,
        "capsule_id" to capsuleId,
        "kind" to kind,
        "created_at" to createdAt,
        "members" to members.toSortedMap().mapValues {
            mapOf("sha256" to it.value.sha256, "bytes" to it.value.bytes)
        },
        "details" to details.toSortedMap(),
    )

    val manifestSha256: String get() = canonicalSha256(toCanonicalDict())
}

/**
 * In-memory recovery capsule: member bytes plus the detached manifest.
 * [writeOrder] records the publication order; the manifest is ALWAYS last.
 */
data class RecoveryCapsule(
    val files: Map<String, ByteArray>,
    val manifest: CapsuleManifest,
    val writeOrder: List<String>,
) {
    init {
        require(writeOrder.isNotEmpty() && writeOrder.last() == MANIFEST_MEMBER_NAME) {
            "manifest must be written last"
        }
        require(writeOrder.dropLast(1).toSet() == files.keys) {
            "write order must cover exactly the member files, then the manifest"
        }
    }
}

class RecoveryCapsuleBuilder(
    private val kind: String,
    private val createdAt: String,
) {
    private val files = LinkedHashMap<String, ByteArray>()
    private val details = LinkedHashMap<String, String>()

    fun addFile(path: String, bytes: ByteArray): RecoveryCapsuleBuilder {
        require(path != MANIFEST_MEMBER_NAME) { "manifest is synthesized; do not add it as a member" }
        require(path !in files) { "duplicate capsule member '$path'" }
        files[path] = bytes
        return this
    }

    fun addDetail(key: String, value: String): RecoveryCapsuleBuilder {
        details[key] = value
        return this
    }

    /** Freeze members, then compute and attach the manifest LAST. */
    fun build(): RecoveryCapsule {
        val frozen = files.toMap()
        val members = frozen.mapValues { (_, bytes) ->
            CapsuleMember(sha256 = CanonicalJson.sha256Hex(bytes), bytes = bytes.size.toLong())
        }
        val preManifest = CapsuleManifest(
            capsuleId = "",
            kind = kind,
            createdAt = createdAt,
            members = members,
            details = details.toMap(),
        )
        val capsuleId = "capsule-" + canonicalSha256(
            mapOf("kind" to kind, "created_at" to createdAt, "members" to preManifest.toCanonicalDict()["members"])
        ).take(24)
        val manifest = preManifest.copy(capsuleId = capsuleId)
        return RecoveryCapsule(
            files = frozen,
            manifest = manifest,
            writeOrder = frozen.keys.toList() + MANIFEST_MEMBER_NAME,
        )
    }
}

@Serializable
data class CapsuleVerification(
    val ok: Boolean,
    val mismatches: List<String>,
    @SerialName("checks_passed") val checksPassed: List<String>,
)

/**
 * Fail-closed capsule verifier. Verification order (mirroring
 * cloud_bundle.verify_and_extract doctrine):
 *   1. manifest schema
 *   2. member-set equality (both directions)
 *   3. per-member sha256 + byte count
 *   4. architecture identity
 *   5. candidate generation
 *   6. optimizer policy
 *   7. soul/checkpoint pairing
 *   8. lineage (base != candidate, bundle/checkpoint/pointer chaining)
 * Any mismatch is collected; the capsule is rejected if ANY disagree.
 */
object RecoveryCapsuleVerifier {

    fun verify(
        files: Map<String, ByteArray>,
        manifest: CapsuleManifest,
        expectedArchitectureId: String? = null,
        expectedCandidateGenerationId: String? = null,
    ): CapsuleVerification {
        val mismatches = mutableListOf<String>()
        val passed = mutableListOf<String>()

        // 1. manifest schema
        if (manifest.schema != CAPSULE_MANIFEST_SCHEMA) {
            mismatches += "manifest schema mismatch: '${manifest.schema}' != '$CAPSULE_MANIFEST_SCHEMA'"
            return CapsuleVerification(ok = false, mismatches = mismatches, checksPassed = passed)
        }
        passed += "manifest-schema"

        // 2. member-set equality, both directions
        val declared = manifest.members.keys
        val present = files.keys
        (declared - present).forEach { mismatches += "manifest member '$it' missing from capsule" }
        (present - declared).forEach { mismatches += "capsule member '$it' absent from manifest" }
        if (mismatches.isNotEmpty()) {
            return CapsuleVerification(false, mismatches, passed)
        }
        passed += "member-set"

        // 3. per-member sha256 + bytes
        for ((path, member) in manifest.members.toSortedMap()) {
            val bytes = files.getValue(path)
            if (bytes.size.toLong() != member.bytes) {
                mismatches += "member '$path' byte count ${bytes.size} != manifest ${member.bytes}"
            }
            val actual = CanonicalJson.sha256Hex(bytes)
            if (actual != member.sha256) {
                mismatches += "member '$path' sha256 $actual != manifest ${member.sha256}"
            }
        }
        if (mismatches.isNotEmpty()) {
            return CapsuleVerification(false, mismatches, passed)
        }
        passed += "member-hashes"

        // Structural identity checks require the typed members.
        val anatomy = decode<AnatomyConfig>(files, "anatomy.json", mismatches)
        val soul = decode<SoulSnapshot>(files, "soul.json", mismatches)
        val policy = decode<LearningPolicy>(files, "learning_policy.json", mismatches)
        val checkpoint = decode<CheckpointRecord>(files, "checkpoint.json", mismatches)
        val bundle = decode<AcceptedStepBundle>(files, "accepted_bundle.json", mismatches)
        val tranche = decode<ResourceTranche>(files, "tranche.json", mismatches)
        if (mismatches.isNotEmpty()) {
            return CapsuleVerification(false, mismatches, passed)
        }
        passed += "member-decode"

        // 4. architecture identity
        if (soul!!.architectureId != anatomy!!.architectureId) {
            mismatches += "architecture identity mismatch: soul '${soul.architectureId}' != anatomy '${anatomy.architectureId}'"
        }
        if (expectedArchitectureId != null && anatomy.architectureId != expectedArchitectureId) {
            mismatches += "architecture identity mismatch: capsule '${anatomy.architectureId}' != expected '$expectedArchitectureId'"
        }
        passed += "architecture-identity"

        // 5. candidate generation
        val candidateIds = listOf(
            checkpoint!!.candidateGenerationId,
            bundle!!.candidateGenerationId,
            tranche!!.candidateGenerationId,
        )
        if (candidateIds.toSet().size != 1) {
            mismatches += "candidate generation mismatch across checkpoint/bundle/tranche: $candidateIds"
        }
        if (expectedCandidateGenerationId != null &&
            checkpoint.candidateGenerationId != expectedCandidateGenerationId
        ) {
            mismatches += "candidate generation mismatch: capsule '${checkpoint.candidateGenerationId}' != expected '$expectedCandidateGenerationId'"
        }
        if (checkpoint.baseGenerationId == checkpoint.candidateGenerationId) {
            mismatches += "lineage violation: base_generation_id == candidate_generation_id"
        }
        passed += "candidate-generation"

        // 6. optimizer policy
        if (checkpoint.learningPolicyId != policy!!.policyId) {
            mismatches += "optimizer policy mismatch: checkpoint '${checkpoint.learningPolicyId}' != policy '${policy.policyId}'"
        }
        if (tranche.learningPolicyId != policy.policyId) {
            mismatches += "optimizer policy mismatch: tranche '${tranche.learningPolicyId}' != policy '${policy.policyId}'"
        }
        passed += "optimizer-policy"

        // 7. soul/checkpoint pairing
        if (soul.parameterGeneration != checkpoint.candidateGenerationId) {
            mismatches += "soul/checkpoint pairing mismatch: soul parameter_generation " +
                "'${soul.parameterGeneration}' != checkpoint candidate '${checkpoint.candidateGenerationId}'"
        }
        if (bundle.checkpointId != checkpoint.checkpointId) {
            mismatches += "soul/checkpoint pairing mismatch: bundle checkpoint '${bundle.checkpointId}' != '${checkpoint.checkpointId}'"
        }
        if (bundle.afterSoulId != soul.soulId) {
            mismatches += "soul/checkpoint pairing mismatch: bundle after_soul '${bundle.afterSoulId}' != soul '${soul.soulId}'"
        }
        passed += "soul-checkpoint-pairing"

        // 8. lineage
        if (tranche.planId != checkpoint.planId) {
            mismatches += "lineage mismatch: tranche plan '${tranche.planId}' != checkpoint plan '${checkpoint.planId}'"
        }
        if (checkpoint.step != bundle.step) {
            mismatches += "lineage mismatch: checkpoint step ${checkpoint.step} != bundle step ${bundle.step}"
        }
        passed += "lineage"

        return CapsuleVerification(ok = mismatches.isEmpty(), mismatches = mismatches, checksPassed = passed)
    }

    private inline fun <reified T> decode(
        files: Map<String, ByteArray>,
        path: String,
        mismatches: MutableList<String>,
    ): T? {
        val bytes = files[path]
        if (bytes == null) {
            mismatches += "required member '$path' missing"
            return null
        }
        return try {
            AxonJson.decodeFromString<T>(bytes.decodeToString())
        } catch (e: Exception) {
            mismatches += "member '$path' failed to decode: ${e.message}"
            null
        }
    }
}
