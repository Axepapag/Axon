package com.gliksbot.axonhome.core.domain

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Trainer contracts — mirrors runtime/trainer/ (contracts.py, learning.py,
 * tranche.py, step_bundle.py, lifecycle.py). Field names and schema strings
 * match the canonical Axon JSON records.
 */

const val LEARNING_POLICY_SCHEMA = "axon-trainer-learning-policy-v1"
const val RESOURCE_TRANCHE_SCHEMA = "axon-trainer-resource-tranche-v1"
const val TRANCHE_CONTINUATION_SCHEMA = "axon-trainer-tranche-continuation-v1"
const val ACCEPTED_STEP_BUNDLE_SCHEMA = "axon-accepted-reasoning-training-step-v1"
const val ACCEPTED_STEP_POINTER_SCHEMA = "axon-accepted-training-step-pointer-v1"
const val CHECKPOINT_RECORD_SCHEMA = "axon-trainer-candidate-checkpoint-v2"
const val OPTIMIZATION_STEP_RECEIPT_SCHEMA = "axon-trainer-optimization-step-v2"

/** Rolling accepted-step / checkpoint retention (capacity policy = 3). */
const val ROLLING_RETENTION = 3

@Serializable
enum class CandidateLifecycleStatus {
    @SerialName("prepared") PREPARED,
    @SerialName("running") RUNNING,
    @SerialName("paused") PAUSED,
    @SerialName("completed") COMPLETED,
    @SerialName("rejected") REJECTED,
    @SerialName("gate_passed") GATE_PASSED,
    @SerialName("promotion_proposed") PROMOTION_PROPOSED,
    @SerialName("promoted") PROMOTED,
    @SerialName("retired") RETIRED,
}

@Serializable
enum class OptimizerKind {
    @SerialName("adamw") ADAMW,
    @SerialName("sgd") SGD,
}

@Serializable
enum class SchedulerKind {
    @SerialName("constant") CONSTANT,
    @SerialName("warmup_cosine") WARMUP_COSINE,
}

@Serializable
enum class PrecisionMode {
    @SerialName("fp32") FP32,
    @SerialName("bf16") BF16,
    @SerialName("fp16") FP16,
}

/** Governed learning policy (axon-trainer-learning-policy-v1). */
@Serializable
data class LearningPolicy(
    val optimizer: OptimizerKind = OptimizerKind.ADAMW,
    @SerialName("learning_rate") val learningRate: Double,
    @SerialName("weight_decay") val weightDecay: Double = 0.0,
    @SerialName("adam_beta1") val adamBeta1: Double = 0.9,
    @SerialName("adam_beta2") val adamBeta2: Double = 0.999,
    @SerialName("adam_eps") val adamEps: Double = 1e-8,
    @SerialName("sgd_momentum") val sgdMomentum: Double? = null,
    val scheduler: SchedulerKind = SchedulerKind.CONSTANT,
    @SerialName("warmup_steps") val warmupSteps: Int = 0,
    @SerialName("min_lr_ratio") val minLrRatio: Double = 0.0,
    @SerialName("schedule_steps") val scheduleSteps: Int = 0,
    @SerialName("gradient_accumulation_steps") val gradientAccumulationSteps: Int = 1,
    @SerialName("gradient_clip_norm") val gradientClipNorm: Double = 1.0,
    @SerialName("max_gradient_l2") val maxGradientL2: Double,
    @SerialName("max_update_l2") val maxUpdateL2: Double,
    val precision: PrecisionMode = PrecisionMode.FP32,
    @SerialName("exact_scope_verification_each_step") val exactScopeVerificationEachStep: Boolean = true,
    @SerialName("full_parameter_telemetry_each_step") val fullParameterTelemetryEachStep: Boolean = false,
    @SerialName("objective_program_id") val objectiveProgramId: String,
    @SerialName("schema") val schema: String = LEARNING_POLICY_SCHEMA,
) {
    init {
        require(schema == LEARNING_POLICY_SCHEMA) { "unsupported learning policy schema '$schema'" }
        require(learningRate > 0.0) { "learning_rate must be positive" }
        require(gradientAccumulationSteps >= 1) { "gradient_accumulation_steps must be >= 1" }
    }

    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to LEARNING_POLICY_SCHEMA,
        "optimizer" to AxonJson.encodeToString(OptimizerKind.serializer(), optimizer).trim('"'),
        "learning_rate" to learningRate,
        "weight_decay" to weightDecay,
        "adam_beta1" to adamBeta1,
        "adam_beta2" to adamBeta2,
        "adam_eps" to adamEps,
        "sgd_momentum" to sgdMomentum,
        "scheduler" to AxonJson.encodeToString(SchedulerKind.serializer(), scheduler).trim('"'),
        "warmup_steps" to warmupSteps,
        "min_lr_ratio" to minLrRatio,
        "schedule_steps" to scheduleSteps,
        "gradient_accumulation_steps" to gradientAccumulationSteps,
        "gradient_clip_norm" to gradientClipNorm,
        "max_gradient_l2" to maxGradientL2,
        "max_update_l2" to maxUpdateL2,
        "precision" to AxonJson.encodeToString(PrecisionMode.serializer(), precision).trim('"'),
        "exact_scope_verification_each_step" to exactScopeVerificationEachStep,
        "full_parameter_telemetry_each_step" to fullParameterTelemetryEachStep,
        "objective_program_id" to objectiveProgramId,
    )

    val policyId: String get() = canonicalSha256(toCanonicalDict())
}

/** Renewable resource budget for one candidate rung of training. */
@Serializable
data class ResourceTranche(
    @SerialName("module_id") val moduleId: String,
    @SerialName("candidate_generation_id") val candidateGenerationId: String,
    @SerialName("plan_id") val planId: String,
    @SerialName("learning_policy_id") val learningPolicyId: String,
    @SerialName("base_global_step") val baseGlobalStep: Int,
    val steps: Int,
    @SerialName("parent_bundle_id") val parentBundleId: String?,
    val purpose: String,
    @SerialName("schema") val schema: String = RESOURCE_TRANCHE_SCHEMA,
) {
    init {
        require(schema == RESOURCE_TRANCHE_SCHEMA) { "unsupported tranche schema '$schema'" }
        require(steps >= 1) { "tranche steps must be >= 1" }
        require(baseGlobalStep >= 0) { "base_global_step must be non-negative" }
    }

    val finalGlobalStep: Int get() = baseGlobalStep + steps

    fun admits(globalStep: Int): Boolean = globalStep in baseGlobalStep until finalGlobalStep

    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to RESOURCE_TRANCHE_SCHEMA,
        "module_id" to moduleId,
        "candidate_generation_id" to candidateGenerationId,
        "plan_id" to planId,
        "learning_policy_id" to learningPolicyId,
        "base_global_step" to baseGlobalStep,
        "steps" to steps,
        "parent_bundle_id" to parentBundleId,
        "purpose" to purpose,
    )

    val trancheId: String get() = canonicalSha256(toCanonicalDict())
}

@Serializable
data class TrancheContinuation(
    @SerialName("parent_bundle_id") val parentBundleId: String,
    @SerialName("parent_checkpoint_id") val parentCheckpointId: String,
    @SerialName("parent_optimizer_receipt_id") val parentOptimizerReceiptId: String,
    @SerialName("parent_soul_id") val parentSoulId: String,
    @SerialName("parent_global_step") val parentGlobalStep: Int,
    @SerialName("prior_tranche_id") val priorTrancheId: String,
    @SerialName("schema") val schema: String = TRANCHE_CONTINUATION_SCHEMA,
) {
    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to TRANCHE_CONTINUATION_SCHEMA,
        "parent_bundle_id" to parentBundleId,
        "parent_checkpoint_id" to parentCheckpointId,
        "parent_optimizer_receipt_id" to parentOptimizerReceiptId,
        "parent_soul_id" to parentSoulId,
        "parent_global_step" to parentGlobalStep,
        "prior_tranche_id" to priorTrancheId,
    )

    val continuationId: String get() = canonicalSha256(toCanonicalDict())
}

@Serializable
data class OptimizationStepReceipt(
    val step: Int,
    @SerialName("micro_step") val microStep: Int,
    val loss: Double,
    @SerialName("gradient_l2") val gradientL2: Double,
    @SerialName("gradient_clip_norm") val gradientClipNorm: Double,
    @SerialName("learning_rate") val learningRate: Double,
    @SerialName("weight_decay") val weightDecay: Double,
    @SerialName("precision_mode") val precisionMode: PrecisionMode,
    @SerialName("update_l2") val updateL2: Double,
    @SerialName("telemetry_frame_id") val telemetryFrameId: String,
    @SerialName("changed_tensor_names") val changedTensorNames: List<String>,
    @SerialName("unchanged_tensor_names") val unchangedTensorNames: List<String>,
    @SerialName("schema") val schema: String = OPTIMIZATION_STEP_RECEIPT_SCHEMA,
) {
    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to OPTIMIZATION_STEP_RECEIPT_SCHEMA,
        "step" to step,
        "micro_step" to microStep,
        "loss" to loss,
        "gradient_l2" to gradientL2,
        "gradient_clip_norm" to gradientClipNorm,
        "learning_rate" to learningRate,
        "weight_decay" to weightDecay,
        "precision_mode" to AxonJson.encodeToString(PrecisionMode.serializer(), precisionMode).trim('"'),
        "update_l2" to updateL2,
        "telemetry_frame_id" to telemetryFrameId,
        "changed_tensor_names" to changedTensorNames.toSet().sorted(),
        "unchanged_tensor_names" to unchangedTensorNames.toSet().sorted(),
    )

    val receiptId: String get() = canonicalSha256(toCanonicalDict())
}

@Serializable
data class CheckpointRecord(
    @SerialName("module_id") val moduleId: String,
    @SerialName("base_generation_id") val baseGenerationId: String,
    @SerialName("candidate_generation_id") val candidateGenerationId: String,
    @SerialName("plan_id") val planId: String,
    @SerialName("authorization_id") val authorizationId: String,
    @SerialName("learning_policy_id") val learningPolicyId: String,
    val step: Int,
    @SerialName("micro_step") val microStep: Int,
    @SerialName("accumulation_index") val accumulationIndex: Int,
    @SerialName("parameter_manifest_id") val parameterManifestId: String,
    @SerialName("artifact_relpath") val artifactRelpath: String,
    @SerialName("artifact_sha256") val artifactSha256: String,
    @SerialName("artifact_bytes") val artifactBytes: Long,
    @SerialName("optimizer_included") val optimizerIncluded: Boolean,
    @SerialName("gradient_state_included") val gradientStateIncluded: Boolean,
    @SerialName("scaler_included") val scalerIncluded: Boolean,
    @SerialName("current_learning_rate") val currentLearningRate: Double,
    @SerialName("accumulated_loss_sum") val accumulatedLossSum: Double,
    @SerialName("previous_checkpoint_id") val previousCheckpointId: String?,
    @SerialName("schema") val schema: String = CHECKPOINT_RECORD_SCHEMA,
) {
    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to CHECKPOINT_RECORD_SCHEMA,
        "module_id" to moduleId,
        "base_generation_id" to baseGenerationId,
        "candidate_generation_id" to candidateGenerationId,
        "plan_id" to planId,
        "authorization_id" to authorizationId,
        "learning_policy_id" to learningPolicyId,
        "step" to step,
        "micro_step" to microStep,
        "accumulation_index" to accumulationIndex,
        "parameter_manifest_id" to parameterManifestId,
        "artifact_relpath" to artifactRelpath,
        "artifact_sha256" to artifactSha256,
        "artifact_bytes" to artifactBytes,
        "optimizer_included" to optimizerIncluded,
        "gradient_state_included" to gradientStateIncluded,
        "scaler_included" to scalerIncluded,
        "current_learning_rate" to currentLearningRate,
        "accumulated_loss_sum" to accumulatedLossSum,
        "previous_checkpoint_id" to previousCheckpointId,
    )

    val checkpointId: String get() = canonicalSha256(toCanonicalDict())
}

/** Accepted global step bundle (crash-safe publication unit). */
@Serializable
data class AcceptedStepBundle(
    @SerialName("intent_id") val intentId: String,
    @SerialName("module_id") val moduleId: String,
    @SerialName("candidate_generation_id") val candidateGenerationId: String,
    @SerialName("core_id") val coreId: String,
    val step: Int,
    @SerialName("optimization_receipt_id") val optimizationReceiptId: String,
    @SerialName("checkpoint_id") val checkpointId: String,
    @SerialName("candidate_soul_manifest_id") val candidateSoulManifestId: String,
    @SerialName("before_soul_id") val beforeSoulId: String,
    @SerialName("after_soul_id") val afterSoulId: String,
    @SerialName("soul_receipt_ids") val soulReceiptIds: List<String>,
    @SerialName("previous_bundle_id") val previousBundleId: String?,
    @SerialName("schema") val schema: String = ACCEPTED_STEP_BUNDLE_SCHEMA,
) {
    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to ACCEPTED_STEP_BUNDLE_SCHEMA,
        "intent_id" to intentId,
        "module_id" to moduleId,
        "candidate_generation_id" to candidateGenerationId,
        "core_id" to coreId,
        "step" to step,
        "optimization_receipt_id" to optimizationReceiptId,
        "checkpoint_id" to checkpointId,
        "candidate_soul_manifest_id" to candidateSoulManifestId,
        "before_soul_id" to beforeSoulId,
        "after_soul_id" to afterSoulId,
        "soul_receipt_ids" to soulReceiptIds.sorted(),
        "previous_bundle_id" to previousBundleId,
    )

    val bundleId: String get() = canonicalSha256(toCanonicalDict())
}

@Serializable
data class AcceptedStepPointer(
    @SerialName("module_id") val moduleId: String,
    @SerialName("candidate_generation_id") val candidateGenerationId: String,
    @SerialName("current_bundle_id") val currentBundleId: String,
    @SerialName("current_step") val currentStep: Int,
    @SerialName("rolling_bundle_ids") val rollingBundleIds: List<String>,
    @SerialName("schema") val schema: String = ACCEPTED_STEP_POINTER_SCHEMA,
) {
    init {
        require(rollingBundleIds.size <= ROLLING_RETENTION) {
            "rolling bundle ids exceed retention $ROLLING_RETENTION"
        }
        require(currentBundleId in rollingBundleIds) {
            "current bundle must be inside the rolling window"
        }
    }

    fun toCanonicalDict(): Map<String, Any?> = mapOf(
        "schema" to ACCEPTED_STEP_POINTER_SCHEMA,
        "module_id" to moduleId,
        "candidate_generation_id" to candidateGenerationId,
        "current_bundle_id" to currentBundleId,
        "current_step" to currentStep,
        "rolling_bundle_ids" to rollingBundleIds,
    )

    val pointerId: String get() = canonicalSha256(toCanonicalDict())
}
