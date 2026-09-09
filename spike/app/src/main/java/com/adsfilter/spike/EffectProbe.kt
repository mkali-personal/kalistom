package com.adsfilter.spike

import android.media.audiofx.AudioEffect
import android.media.audiofx.DynamicsProcessing
import android.media.audiofx.Equalizer
import android.media.audiofx.LoudnessEnhancer

/**
 * Phase 4 spike, pulled forward: can an UNPRIVILEGED app attenuate the global output mix?
 *
 * AudioEffect session 0 means "apply to the output mix" - i.e. everything the device is playing,
 * regardless of which app produced it. If this works it is a far better actuator than
 * setStreamVolume: no visible volume slider movement, no Bluetooth absolute-volume interaction,
 * no dependency on the playing app exposing a MediaSession.
 *
 * Constructing and enabling an effect proves only that the framework ACCEPTED it. Whether the
 * audio actually gets quieter must be judged by ear - hence the hold period.
 */
object EffectProbe {

    private const val OUTPUT_MIX_SESSION = 0

    /** Runs on a background thread. [hold] is invoked while attenuation should be audible. */
    fun run(log: (String) -> Unit, hold: () -> Unit) {
        log("=== global output-mix attenuation probe (session 0) ===")
        log("Audio must be PLAYING. Listen for a volume drop after each ATTACHED line.")
        probeLoudnessEnhancer(log, hold)
        probeEqualizer(log, hold)
        probeDynamicsProcessing(log, hold)
        log("=== probe done ===")
    }

    private fun probeLoudnessEnhancer(log: (String) -> Unit, hold: () -> Unit) {
        var fx: LoudnessEnhancer? = null
        try {
            fx = LoudnessEnhancer(OUTPUT_MIX_SESSION)
            // Negative millibels = attenuation. May be clamped to 0 by the implementation,
            // since this effect is nominally a booster.
            fx.setTargetGain(-4000)
            fx.enabled = true
            log("LoudnessEnhancer  ATTACHED  enabled=" + fx.enabled + "  targetGain=" + fx.targetGain + " mB")
            hold()
        } catch (t: Throwable) {
            log("LoudnessEnhancer  FAILED    " + describe(t))
        } finally {
            release(fx, log, "LoudnessEnhancer")
        }
    }

    private fun probeEqualizer(log: (String) -> Unit, hold: () -> Unit) {
        var fx: Equalizer? = null
        try {
            fx = Equalizer(0, OUTPUT_MIX_SESSION)
            val range = fx.bandLevelRange     // [min, max] in millibels
            val min = range[0]
            for (b in 0 until fx.numberOfBands) {
                fx.setBandLevel(b.toShort(), min)
            }
            fx.enabled = true
            log("Equalizer         ATTACHED  enabled=" + fx.enabled + "  bands=" + fx.numberOfBands + "  floor=" + min + " mB")
            hold()
        } catch (t: Throwable) {
            log("Equalizer         FAILED    " + describe(t))
        } finally {
            release(fx, log, "Equalizer")
        }
    }

    private fun probeDynamicsProcessing(log: (String) -> Unit, hold: () -> Unit) {
        var fx: DynamicsProcessing? = null
        try {
            fx = DynamicsProcessing(OUTPUT_MIX_SESSION)
            fx.setInputGainAllChannelsTo(-60f)   // dB
            fx.enabled = true
            log("DynamicsProcessing ATTACHED enabled=" + fx.enabled + "  inputGain=-60 dB")
            hold()
        } catch (t: Throwable) {
            log("DynamicsProcessing FAILED   " + describe(t))
        } finally {
            release(fx, log, "DynamicsProcessing")
        }
    }

    private fun release(fx: AudioEffect?, log: (String) -> Unit, name: String) {
        if (fx == null) return
        try {
            fx.enabled = false
            fx.release()
        } catch (t: Throwable) {
            log(name + " release failed: " + describe(t))
        }
    }

    private fun describe(t: Throwable): String {
        val base = t.javaClass.simpleName + ": " + (t.message ?: "(no message)")
        return when (t) {
            is UnsupportedOperationException -> base + "  -> effect not available on this device"
            is IllegalArgumentException -> base + "  -> session 0 likely refused for unprivileged apps"
            is IllegalStateException -> base + "  -> framework refused the attach"
            is SecurityException -> base + "  -> permission denied (needs privileged app)"
            else -> base
        }
    }
}
