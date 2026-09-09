package com.adsfilter

import android.media.audiofx.AudioEffect
import android.media.audiofx.DynamicsProcessing
import android.media.audiofx.LoudnessEnhancer
import android.util.Log

/**
 * Global output-mix attenuation - the Phase 4 actuator, proven attachable in Phase 0.
 *
 * An AudioEffect on session 0 applies to everything the device plays, whichever app produced it,
 * needing only MODIFY_AUDIO_SETTINGS. That is why it beats setStreamVolume: no visible slider
 * movement, no Bluetooth absolute-volume interaction, no dependency on the playing app exposing a
 * MediaSession.
 *
 * CAUTION: it attenuates *everything*, alarms and navigation included. Any caller must guarantee
 * release - see [releaseQuietly] - and Phase 4 will need a watchdog on top.
 */
object Attenuator {

    private const val TAG = "Attenuator"
    private const val OUTPUT_MIX_SESSION = 0

    enum class Kind { LOUDNESS, DYNAMICS }

    @Volatile
    private var effect: AudioEffect? = null

    @Volatile
    var attachedKind: Kind? = null
        private set

    val isAttached: Boolean get() = effect != null

    /** Attaches an attenuating effect to the output mix. Returns false if the framework refused. */
    @Synchronized
    fun attach(kind: Kind, db: Float): Boolean {
        releaseQuietly()
        return try {
            val fx: AudioEffect = when (kind) {
                Kind.LOUDNESS -> LoudnessEnhancer(OUTPUT_MIX_SESSION).apply {
                    setTargetGain((db * 100).toInt())   // millibels
                }
                Kind.DYNAMICS -> DynamicsProcessing(OUTPUT_MIX_SESSION).apply {
                    setInputGainAllChannelsTo(db)
                }
            }
            fx.enabled = true
            effect = fx
            attachedKind = kind
            Log.i(TAG, "attached $kind at $db dB (enabled=${fx.enabled})")
            true
        } catch (t: Throwable) {
            Log.w(TAG, "attach $kind failed: ${t.javaClass.simpleName}: ${t.message}")
            effect = null
            attachedKind = null
            false
        }
    }

    @Synchronized
    fun releaseQuietly() {
        val fx = effect ?: return
        effect = null
        attachedKind = null
        try {
            fx.enabled = false
            fx.release()
            Log.i(TAG, "released")
        } catch (t: Throwable) {
            Log.w(TAG, "release failed: ${t.message}")
        }
    }
}
