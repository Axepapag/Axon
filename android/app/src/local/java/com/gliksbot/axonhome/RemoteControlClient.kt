package com.gliksbot.axonhome

import java.net.HttpURLConnection
import java.net.URL
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

/** Minimal read-mostly client for the cloud VM control plane. */
class RemoteControlClient(
    private val enrollment: RemoteConnectionStore.Enrollment,
) {
    data class RuntimeSummary(
        val controlStatus: String,
        val heartbeatSequence: Long?,
        val tickSequence: Long?,
        val fieldId: String?,
        val trainerStatus: String,
        val candidateGeneration: String?,
        val trainingStep: Long?,
        val loss: Double?,
        val trainingProgressStatus: String?,
        val raw: String,
    )

    suspend fun health(): Boolean = withContext(Dispatchers.IO) {
        request("/v1/health").optJSONObject("control_plane")?.optString("status") == "ok"
    }

    suspend fun runtimeSummary(): RuntimeSummary = withContext(Dispatchers.IO) {
        val root = request("/v1/runtime/summary")
        val heart = root.optJSONObject("heart")
        val head = root.optJSONObject("field_head")
        val trainer = root.optJSONObject("trainer")
        val payload = trainer?.optJSONObject("payload")
        val inspection = payload?.optJSONObject("inspection")
        val lifecycle = inspection?.optJSONObject("latest_lifecycle")
        val step = inspection?.optJSONObject("latest_optimization_step")
        val progress = root.optJSONObject("training_progress")
        RuntimeSummary(
            controlStatus = if (heart == null) "degraded" else "ok",
            heartbeatSequence = heart?.optLongOrNull("heartbeat_sequence"),
            tickSequence = heart?.optLongOrNull("tick_sequence"),
            fieldId = head?.optString("field_id")?.ifBlank { heart?.optString("head_field_id") },
            trainerStatus = lifecycle?.optString("status")?.ifBlank { "idle" } ?: "idle",
            candidateGeneration = lifecycle?.optString("candidate_generation_id")?.ifBlank { null },
            trainingStep = step?.optLongOrNull("step"),
            loss = step?.optDoubleOrNull("loss"),
            trainingProgressStatus = progress?.optString("status")?.ifBlank { null },
            raw = root.toString(2),
        )
    }

    private fun request(path: String): JSONObject {
        val connection = (URL(enrollment.baseUrl + path).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 8_000
            readTimeout = 12_000
            setRequestProperty("Authorization", "Bearer ${enrollment.token}")
            setRequestProperty("Accept", "application/json")
        }
        try {
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val body = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (code !in 200..299) error("HTTP $code: $body")
            return JSONObject(body)
        } finally {
            connection.disconnect()
        }
    }

    private fun JSONObject.optLongOrNull(name: String): Long? =
        if (!has(name) || isNull(name)) null else optLong(name)

    private fun JSONObject.optDoubleOrNull(name: String): Double? =
        if (!has(name) || isNull(name)) null else optDouble(name)
}
