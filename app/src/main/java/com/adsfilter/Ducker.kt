package com.adsfilter

import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import kotlin.math.abs

/**
 * What actually turns the volume down, and - more importantly - what guarantees it comes back up.
 *
 * [Attenuator] is the primitive: a session-0 AudioEffect proven in Phase 0 to attenuate the whole
 * output mix from an unprivileged app. This adds the two things a detector must not be trusted to
 * get right by itself.
 *
 * A FADE. The effect can be stepped, so the gain ramps over [FADE_MS] instead of jumping. A hard
 * cut to silence clicks, and it makes every false positive maximally irritating; a 200 ms ramp on
 * a one-second mistake is something you can fail to notice. It costs nothing in latency terms -
 * 200 ms against a detector that is already several seconds behind the start of a break.
 *
 * A WATCHDOG, which is the part that matters. This effect attenuates *everything*: alarms,
 * navigation, the other half of a phone call. A detector wedged in the muted state, a crashed
 * capture thread or a killed process must not leave the device quiet, so:
 *
 *   - every request carries a deadline, and the ducker releases itself if no one renews it within
 *     [STALE_MS] - a detector that stops calling cannot hold the mute;
 *   - [MAX_CONTINUOUS_MS] caps any single mute regardless of what the detector believes, because
 *     no ad break is eleven minutes long and something is wrong if one appears to be;
 *   - [releaseNow] is idempotent and safe from any thread, so the service's onDestroy and the
 *     capture loop's finally-block can both call it.
 *
 * The failure mode this is designed around is not "the model is wrong" - that costs a few seconds
 * of programme. It is "the phone is silent and the user does not know why", which costs their trust
 * in the whole thing.
 */
object Ducker {

    private const val TAG = "Ducker"

    /** Ramp length. Long enough not to click, short enough not to read as a fault. */
    private const val FADE_MS = 200L
    private const val FADE_STEP_MS = 20L

    /** How long the detector's last request stays valid without renewal. Four frames. */
    private const val STALE_MS = 2_000L

    /** No single mute may outlast this, whatever the detector says. */
    private const val MAX_CONTINUOUS_MS = 11 * 60 * 1000L

    /** Depth of the duck. Not silence: -40 dB is inaudible over anything but leaves a thread of
     * signal, so a mistake sounds like a fault you can hear through rather than a dead phone. */
    private const val TARGET_DB = -40f

    private val thread = HandlerThread("ducker").apply { start() }
    private val handler = Handler(thread.looper)

    @Volatile private var wanted = false
    @Volatile private var deadlineMs = 0L
    @Volatile private var mutedSinceMs = 0L
    private var currentDb = 0f

    /** True while any attenuation is applied. */
    val isDucking: Boolean get() = Attenuator.isAttached

    /** Current attenuation in dB, 0 when clear. Notification copy only. */
    val depthDb: Float get() = currentDb

    /**
     * The detector's standing request, renewed every frame. `true` ducks, `false` releases; either
     * way the deadline is refreshed, so *calling at all* is what keeps the watchdog quiet.
     */
    fun request(duck: Boolean) {
        val now = System.currentTimeMillis()
        deadlineMs = now + STALE_MS
        if (duck && !wanted) mutedSinceMs = now
        if (duck != wanted) {
            wanted = duck
            Log.i(TAG, if (duck) "duck requested" else "release requested")
        }
        handler.post { step() }
    }

    /** Drops everything immediately, from any thread. Safe to call when not ducking. */
    fun releaseNow(reason: String) {
        wanted = false
        handler.post {
            if (Attenuator.isAttached || currentDb != 0f) Log.i(TAG, "release now: $reason")
            currentDb = 0f
            Attenuator.releaseQuietly()
        }
    }

    private fun step() {
        val now = System.currentTimeMillis()

        // Watchdog, checked before anything else so a stuck caller cannot skip it.
        if (wanted && now > deadlineMs) {
            Log.w(TAG, "stale request (${now - deadlineMs} ms past deadline) - releasing")
            wanted = false
        }
        if (wanted && mutedSinceMs != 0L && now - mutedSinceMs > MAX_CONTINUOUS_MS) {
            Log.w(TAG, "mute exceeded ${MAX_CONTINUOUS_MS / 1000}s - releasing, something is wrong")
            wanted = false
        }

        val target = if (wanted) TARGET_DB else 0f
        if (abs(currentDb - target) < 0.51f) {
            currentDb = target
            if (target == 0f) {
                Attenuator.releaseQuietly()
                mutedSinceMs = 0L
            }
            // Keep one timer alive while ducking, so the watchdog fires even if the detector dies.
            if (wanted) handler.postDelayed({ step() }, STALE_MS / 2)
            return
        }

        val stepDb = (TARGET_DB / (FADE_MS / FADE_STEP_MS.toFloat()))
        currentDb = if (target < currentDb) {
            maxOf(currentDb + stepDb, target)
        } else {
            minOf(currentDb - stepDb, target)
        }
        if (!Attenuator.attach(Attenuator.Kind.DYNAMICS, currentDb)) {
            // DynamicsProcessing is the deeper of the two but not guaranteed; fall back once.
            if (!Attenuator.attach(Attenuator.Kind.LOUDNESS, currentDb)) {
                Log.w(TAG, "no attenuator would attach - giving up on this duck")
                wanted = false
                currentDb = 0f
                return
            }
        }
        handler.postDelayed({ step() }, FADE_STEP_MS)
    }
}
