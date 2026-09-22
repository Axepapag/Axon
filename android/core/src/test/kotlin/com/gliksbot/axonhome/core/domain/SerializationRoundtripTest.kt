package com.gliksbot.axonhome.core.domain

import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Round-trip every schema-bearing type through the wire codec and confirm
 * the schema strings and field names match the canonical Axon contracts.
 */
class SerializationRoundtripTest {

    private inline fun <reified T> roundtrip(value: T, expectedSchema: String? = null): T {
        val text = AxonJson.encodeToString(value)
        if (expectedSchema != null) {
            val schema = Json.parseToJsonElement(text).let {
                it as kotlinx.serialization.json.JsonObject
            }["schema"]?.let { s -> (s as kotlinx.serialization.json.JsonPrimitive).content }
            assertEquals(expectedSchema, schema)
        }
        return AxonJson.decodeFromString(text)
    }

    @Test
    fun `shared field snapshot roundtrip`() {
        val snapshot = SharedFieldSnapshot.fromTexts(
            mapOf(LogicalRegion.SCRATCH to "abc", LogicalRegion.USER_INPUT to "hi")
        )
        val decoded = roundtrip(snapshot, "shared-field-v4")
        assertEquals(snapshot, decoded)
        assertEquals(snapshot.fieldId, decoded.fieldId)
    }

    @Test
    fun `field delta with all op kinds roundtrips with op discriminator`() {
        val base = SharedFieldSnapshot.fromTexts(mapOf(LogicalRegion.SCRATCH to "0123456789"))
        val delta = FieldDelta(
            baseFieldId = base.fieldId, baseTickId = base.tickId,
            authorCoreId = "sim-core-1", passId = "consolidated",
            operations = listOf(
                InsertText(LogicalRegion.SCRATCH, 0, "x"),
                DeleteText(LogicalRegion.SCRATCH, 1, 2),
                ReplaceText(LogicalRegion.SCRATCH, 3, 4, "y"),
            ),
        )
        val text = AxonJson.encodeToString(delta)
        // Polymorphic discriminator is literally "op" like the Python dicts.
        listOf("\"op\":\"insert\"", "\"op\":\"delete\"", "\"op\":\"replace\"").forEach {
            kotlin.test.assertTrue(text.contains(it), "missing $it in $text")
        }
        kotlin.test.assertTrue(text.contains("\"base_field_id\""))
        val decoded = AxonJson.decodeFromString<FieldDelta>(text)
        assertEquals(delta, decoded)
        assertEquals(delta.deltaId, decoded.deltaId)
    }

    @Test
    fun `english proposal roundtrip`() {
        val base = SharedFieldSnapshot.empty()
        val proposal = EnglishProposal(base.fieldId, 0, "sim-core-1", "first", 64, "proposal text")
        val decoded = roundtrip(proposal, "axon-english-proposal-v1")
        assertEquals(proposal, decoded)
        assertEquals(proposal.proposalId, decoded.proposalId)
    }

    @Test
    fun `technical final verdict roundtrip`() {
        val base = SharedFieldSnapshot.empty()
        val verdict = TechnicalFinalVerdict(base.fieldId, 0, "sim-core-1", 64, "#scratch# n")
        val decoded = roundtrip(verdict, "axon-tagged-final-verdict-v2")
        assertEquals(verdict, decoded)
        assertEquals(verdict.verdictId, decoded.verdictId)
    }

    @Test
    fun `soul snapshot and receipt roundtrip`() {
        val soul = genesisSoul("sim-core-1", "living-d64-english-x", "untrained", "s")
        val decoded = roundtrip(soul, "axon-private-soul-snapshot-v1")
        assertEquals(soul, decoded)
        assertEquals(soul.soulId, decoded.soulId)

        val transition = SoulTransition(soul.coreId, "live", soul.soulId, "first", listOf(SoulTemperature.HOT))
        val (_, receipt) = applySoulTransition(soul, transition)
        val decodedReceipt = roundtrip(receipt, "axon-private-soul-commit-receipt-v1")
        assertEquals(receipt, decodedReceipt)
        assertEquals(receipt.receiptId, decodedReceipt.receiptId)
    }

    @Test
    fun `trainer records roundtrip`() {
        val policy = LearningPolicy(
            learningRate = 1e-4, maxGradientL2 = 10.0, maxUpdateL2 = 1.0,
            objectiveProgramId = "teacher_forced_decode:sequence_cross_entropy:head0",
        )
        assertEquals(policy, roundtrip(policy, "axon-trainer-learning-policy-v1"))

        val tranche = ResourceTranche("m", "cand", "plan", policy.policyId, 0, 4, null, "p")
        assertEquals(tranche, roundtrip(tranche, "axon-trainer-resource-tranche-v1"))

        val continuation = TrancheContinuation("b", "c", "r", "s", 4, tranche.trancheId)
        assertEquals(continuation, roundtrip(continuation, "axon-trainer-tranche-continuation-v1"))

        val receipt = OptimizationStepReceipt(
            step = 1, microStep = 1, loss = 1.5, gradientL2 = 0.4, gradientClipNorm = 1.0,
            learningRate = 1e-4, weightDecay = 0.0, precisionMode = PrecisionMode.FP32,
            updateL2 = 0.01, telemetryFrameId = "t", changedTensorNames = listOf("a"),
            unchangedTensorNames = listOf("b"),
        )
        assertEquals(receipt, roundtrip(receipt, "axon-trainer-optimization-step-v2"))

        val checkpoint = CheckpointRecord(
            moduleId = "m", baseGenerationId = "b", candidateGenerationId = "c",
            planId = "p", authorizationId = "a", learningPolicyId = policy.policyId,
            step = 1, microStep = 1, accumulationIndex = 0, parameterManifestId = "pm",
            artifactRelpath = "checkpoints/x.pt", artifactSha256 = "h", artifactBytes = 10,
            optimizerIncluded = true, gradientStateIncluded = false, scalerIncluded = false,
            currentLearningRate = 1e-4, accumulatedLossSum = 1.5, previousCheckpointId = null,
        )
        assertEquals(checkpoint, roundtrip(checkpoint, "axon-trainer-candidate-checkpoint-v2"))

        val bundle = AcceptedStepBundle(
            intentId = "i", moduleId = "m", candidateGenerationId = "c", coreId = "sim-core-1",
            step = 1, optimizationReceiptId = receipt.receiptId, checkpointId = checkpoint.checkpointId,
            candidateSoulManifestId = "sm", beforeSoulId = "b0", afterSoulId = "b1",
            soulReceiptIds = listOf("r1"), previousBundleId = null,
        )
        assertEquals(bundle, roundtrip(bundle, "axon-accepted-reasoning-training-step-v1"))

        val pointer = AcceptedStepPointer("m", "c", bundle.bundleId, 1, listOf(bundle.bundleId))
        assertEquals(pointer, roundtrip(pointer, "axon-accepted-training-step-pointer-v1"))
    }

    @Test
    fun `health and mask state roundtrip`() {
        val health = HeartHealth(
            epochId = "e", heartbeatSequence = 1, tickSequence = 1, headFieldId = "f",
            headTickId = 1, tickInFlight = false, maskRevision = 0, maskStateId = "m",
            lastViewId = "v", lastFailure = null,
        )
        assertEquals(health, roundtrip(health, "axon-heart-health-v2"))

        val masks = HeartRegionMaskState.defaults()
        assertEquals(masks, roundtrip(masks, "axon-heart-region-masks-v3"))
    }

    @Test
    fun `core descriptor binding and anatomy roundtrip`() {
        val anatomy = AnatomyConfig(nHeads = 2, nLayers = 3, ffnDim = 65536, stateTokens = 8, pageSize = 32)
        assertEquals(anatomy, roundtrip(anatomy))
        val descriptor = CoreDescriptor(coreId = "sim-core-1", architectureId = anatomy.architectureId)
        assertEquals(descriptor, roundtrip(descriptor))
        val binding = CoreBinding(anatomy.architectureId, "untrained", "none", "soul", "runtime")
        assertEquals(binding, roundtrip(binding))
    }

    @Test
    fun `mirror envelope roundtrip carries freshness`() {
        val envelope = MirrorEnvelope(
            data = SharedFieldSnapshot.empty(),
            freshness = StateFreshness.CACHED,
            asOf = "2026-02-16T00:00:00Z",
            source = "SIMULATION",
        )
        val text = AxonJson.encodeToString(envelope)
        kotlin.test.assertTrue(text.contains("\"freshness\":\"CACHED\""))
        val decoded = AxonJson.decodeFromString<MirrorEnvelope<SharedFieldSnapshot>>(text)
        assertEquals(envelope, decoded)
        assertEquals(StateFreshness.CACHED, decoded.freshness)
    }
}
