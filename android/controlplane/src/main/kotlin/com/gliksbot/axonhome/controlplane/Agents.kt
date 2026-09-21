package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** AgentProvider capability advertisement (arch §7.1). */
@Serializable
data class AgentCapabilities(
    val supportsStreaming: Boolean = false,
    val supportsTools: Boolean = false,
    val supportsSystemPrompt: Boolean = false,
    val supportsModelList: Boolean = false,
)

/**
 * Engineering agent registry record (arch §5.5, §10.Q/R). NEVER carries
 * secrets: auth material is a vault key REFERENCE, never a raw key.
 */
@Serializable
data class AgentRecord(
    @SerialName("agent_id") val agentId: String,
    val name: String,
    /** AgentProvider adapter id (e.g. openai-compatible, generic-rest). */
    val provider: String,
    val model: String? = null,
    val online: Boolean,
    val role: String? = null,
    @SerialName("active_task") val activeTask: String? = null,
    val capabilities: AgentCapabilities = AgentCapabilities(),
    /** REFERENCE to a vault entry; never a raw key. */
    @SerialName("vault_key_ref") val vaultKeyRef: String? = null,
)

/** GET /v1/agents. */
@Serializable
data class AgentList(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val agents: List<AgentRecord>,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-agent-list-v1"
    }
}

/** POST /v1/agents. */
@Serializable
data class RegisterAgentRequest(
    val schema: String = SCHEMA,
    val name: String,
    val provider: String,
    @SerialName("base_url") val baseUrl: String? = null,
    val model: String? = null,
    val role: String? = null,
    @SerialName("vault_key_ref") val vaultKeyRef: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-agent-register-v1"
    }
}

/** Routed message record. Secrets are never carried. */
@Serializable
data class AgentMessage(
    val schema: String = SCHEMA,
    @SerialName("message_id") val messageId: String,
    @SerialName("agent_id") val agentId: String,
    /** Sender identity (operator or agent id). */
    val from: String,
    val body: String,
    @SerialName("sent_at") val sentAt: Instant,
    @SerialName("in_reply_to") val inReplyTo: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-agent-message-v1"
    }
}

/** POST /v1/agents/{agentId}/messages request. */
@Serializable
data class AgentMessageRequest(
    val schema: String = SCHEMA,
    val body: String,
    @SerialName("in_reply_to") val inReplyTo: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-agent-message-request-v1"
    }
}
