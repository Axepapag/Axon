package com.gliksbot.axonhome.phone

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.gliksbot.axonhome.core.domain.*
import com.gliksbot.axonhome.core.phone.*
import java.util.UUID

/** One local authoritative body; commits and their audit records share one SQLite transaction. */
class PhoneStore(context: Context) : SQLiteOpenHelper(context, "axon-phone.db", null, 1) {
    private val identity = context.assets.open("axon-identity.txt").bufferedReader().use { it.readText() }
    override fun onConfigure(db: SQLiteDatabase) {
        db.execSQL("PRAGMA synchronous=FULL")
    }
    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("CREATE TABLE body (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
        db.execSQL("CREATE TABLE history (seq INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
        val initial = PhoneState(field = SharedFieldSnapshot.fromTexts(
            mapOf(LogicalRegion.IDENTITY to identity), source = "axon-identity-v1-2026-08-29",
            provenance = "63f7b61587647b991e0b2ded10345dfeb8429539595a6aeb643ada9c7d7049fc"))
        write(db, initial, "initialize")
    }
    override fun onUpgrade(db: SQLiteDatabase, old: Int, new: Int) {
        error("Unsupported state migration: $old -> $new")
    }
    private fun read(db: SQLiteDatabase): PhoneState = db.rawQuery("SELECT payload, sha256 FROM body WHERE id=1", null).use {
        check(it.moveToFirst()) { "Canonical body missing" }
        val payload = it.getString(0)
        check(CanonicalJson.sha256Hex(payload.encodeToByteArray()) == it.getString(1)) { "Canonical body checksum mismatch" }
        AxonJson.decodeFromString(PhoneState.serializer(), payload)
    }
    private fun write(db: SQLiteDatabase, state: PhoneState, action: String) {
        val payload = AxonJson.encodeToString(PhoneState.serializer(), state)
        val hash = CanonicalJson.sha256Hex(payload.encodeToByteArray())
        val row = ContentValues().apply { put("id", 1); put("payload", payload); put("sha256", hash) }
        db.insertWithOnConflict("body", null, row, SQLiteDatabase.CONFLICT_REPLACE).also { check(it != -1L) }
        val audit = ContentValues().apply { put("action", action); put("payload", payload); put("sha256", hash) }
        check(db.insertOrThrow("history", null, audit) != -1L)
    }
    @Synchronized fun state(): PhoneState = read(readableDatabase)
    @Synchronized private fun commit(action: String, transform: (PhoneState) -> PhoneState): PhoneState {
        val db = writableDatabase
        db.beginTransaction()
        try {
            val next = transform(read(db))
            write(db, next, action)
            db.setTransactionSuccessful()
            return next
        } finally { db.endTransaction() }
    }
    fun ingress(text: String) = commit("user_ingress") { PhoneHeart.ingress(it, text, UUID.randomUUID().toString()) }
    fun pause(value: Boolean) = commit(if (value) "pause" else "resume") { it.copy(paused = value) }
    fun mask(region: LogicalRegion, percent: Int) = commit("mask:${region.wireName}:$percent") { PhoneHeart.mask(it, region, percent) }
    fun restore(capsule: PhoneRecovery): PhoneState {
        val verified = capsule.verify()
        require(verified.field.region(LogicalRegion.IDENTITY).text == identity) { "Identity differs; governed migration required" }
        // Prior body remains in history. Restore is explicit and recorded atomically.
        return commit("restore:${capsule.sha256}") { verified }
    }
}
