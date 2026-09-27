package com.adsfilter

import android.content.Context
import android.util.Log
import org.json.JSONObject

/**
 * The trained head, running on the phone: embedding stream in, mute decision out.
 *
 * Deliberately the dumbest part of the system. YAMNet does the listening and the desktop does the
 * learning; this is one dot product, a running average and two thresholds. Everything it needs
 * arrives in head_weights.json, which the trainer writes - including the thresholds, so tuning
 * never means editing Kotlin.
 *
 * WHY THE ORDER OF OPERATIONS MATTERS. This has to reproduce trainer/train.py exactly, because a
 * mismatch here does not crash, it silently degrades: the model would be scoring vectors assembled
 * differently from the ones it was fitted on. Three places where it is easy to get wrong, all of
 * them checked by trainer/parity_check.py against a real session:
 *
 *  1. Column order. The base frame is 1024 embedding values followed by ONE loudness value,
 *     scaled (dBFS + 60) / 60 - the same expression as in train.py.
 *  2. Tap order. The stacked vector is oldest frame first: [frame i-9, i-8, ... , i]. That is
 *     `stack_context`'s `arange(n)[::-1]`, and reversing it would still produce a plausible
 *     number from the wrong weights.
 *  3. Edges. Training clamped at each recording's start, repeating the first frame rather than
 *     dropping rows. A session here is a recording, so the first frames repeat frame 0 too.
 *
 * WHY A RUNNING AVERAGE OF THE SCORE, AND NOT A VOTE. Measured on held-out audio, averaging the
 * raw scores over ~5 s and then thresholding removes 441 ad-seconds per hour; rounding each frame
 * to yes/no first and counting the yeses removes 326, and switches the mute three times as often.
 * Rounding throws away the certainty the average needs - a frame at 0.51 should not weigh as much
 * as one at 0.999. The averaged score is also strictly MORE selective than a single frame (371:1
 * against 152:1 at 0.99), because the head saturates: inside a real break nearly every frame is
 * ~1.0, so the mean is ~1.0, and it only sits in the middle where frames disagree, which is at
 * boundaries and on isolated spikes. Being hesitant exactly there is the point.
 *
 * Note that `on` and `off` therefore apply to the AVERAGE, not to one frame, and are not
 * interchangeable with per-frame thresholds. head_weights.json carries a `smoothing` block so the
 * two cannot be confused.
 */
class Detector private constructor(
    private val contextFrames: Int,
    private val baseDim: Int,
    private val mean: FloatArray,
    private val scale: FloatArray,
    private val weights: FloatArray,
    private val bias: Float,
    private val smoothFrames: Int,
    val onThreshold: Float,
    val offThreshold: Float,
) {

    companion object {
        private const val TAG = "Detector"
        const val ASSET = "head_weights.json"

        /**
         * Loads the head, or returns null if the asset is missing or malformed. Null means "record
         * but never mute", which is the safe failure: the app degrades to what it did before.
         */
        fun load(context: Context): Detector? = try {
            val text = context.assets.open(ASSET).bufferedReader().use { it.readText() }
            val o = JSONObject(text)
            val ctx = o.getInt("context_frames")
            val inputDim = o.getInt("input_dim")
            require(inputDim % ctx == 0) { "input_dim $inputDim is not a multiple of $ctx taps" }
            val base = inputDim / ctx
            require(base == Yamnet.EMBEDDING_DIM + 1) {
                "expected ${Yamnet.EMBEDDING_DIM + 1} base columns, file says $base"
            }
            val smoothing = o.optJSONObject("smoothing")
            val d = Detector(
                contextFrames = ctx,
                baseDim = base,
                mean = floats(o, "mean", inputDim),
                scale = floats(o, "scale", inputDim),
                weights = floats(o, "weights", inputDim),
                bias = o.getDouble("bias").toFloat(),
                smoothFrames = smoothing?.optInt("frames", 1) ?: 1,
                onThreshold = o.getDouble("on").toFloat(),
                offThreshold = o.getDouble("off").toFloat(),
            )
            Log.i(
                TAG,
                "loaded: $ctx taps x $base = $inputDim inputs, smoothing ${d.smoothFrames} frames " +
                    "(${"%.1f".format(d.smoothFrames * Yamnet.HOP.toFloat() / Yamnet.SAMPLE_RATE)}s), " +
                    "on=${d.onThreshold} off=${d.offThreshold}"
            )
            d
        } catch (t: Throwable) {
            Log.w(TAG, "no usable $ASSET (${t.javaClass.simpleName}: ${t.message}) - will not mute")
            null
        }

        private fun floats(o: JSONObject, key: String, expect: Int): FloatArray {
            val a = o.getJSONArray(key)
            require(a.length() == expect) { "$key has ${a.length()} values, expected $expect" }
            return FloatArray(expect) { a.getDouble(it).toFloat() }
        }
    }

    /**
     * weight / scale, folded together once at load. The score needs
     * sum over columns of ((x - mean) / scale) * weight, and scale and weight are both fixed, so
     * the division is a constant - doing it per frame would be 10,250 divisions every 0.48 s for
     * no reason.
     */
    private val invScaleWeight = FloatArray(weights.size) { weights[it] / scale[it] }

    /** The most recent [contextFrames] base frames, oldest-first once full. A ring, not a copy. */
    private val ring = FloatArray(contextFrames * baseDim)
    private var framesSeen = 0L
    private var writeSlot = 0

    private val recent = FloatArray(smoothFrames)
    private var recentCount = 0
    private var recentSlot = 0
    private var recentSum = 0.0

    var rawScore = 0f
        private set
    var smoothedScore = 0f
        private set
    var shouldMute = false
        private set

    fun reset() {
        java.util.Arrays.fill(ring, 0f)
        framesSeen = 0
        writeSlot = 0
        java.util.Arrays.fill(recent, 0f)
        recentCount = 0
        recentSlot = 0
        recentSum = 0.0
        rawScore = 0f
        smoothedScore = 0f
        shouldMute = false
    }

    /**
     * Feeds one frame and returns the mute decision. Called on the capture thread, so it allocates
     * nothing and runs one pass over 10,250 floats - about 10k multiply-adds per 0.48 s, which is
     * noise next to YAMNet itself.
     */
    fun push(embedding: FloatArray, rmsDbfs: Float): Boolean {
        val at = writeSlot * baseDim
        System.arraycopy(embedding, 0, ring, at, Yamnet.EMBEDDING_DIM)
        ring[at + Yamnet.EMBEDDING_DIM] = (rmsDbfs + 60f) / 60f
        writeSlot = (writeSlot + 1) % contextFrames
        framesSeen++

        // Oldest tap first. Before the ring has filled, training clamped to the recording's first
        // frame, so the missing taps read frame 0 rather than zeros - which would be a valid
        // vector of a silence that never happened.
        var acc = bias.toDouble()
        var col = 0
        for (tap in 0 until contextFrames) {
            val lag = contextFrames - 1 - tap
            // Frame k lives in slot k % contextFrames. Clamping only happens while
            // framesSeen <= lag < contextFrames, so frame 0 is still in slot 0 and unoverwritten.
            val slot = if (framesSeen > lag) ((framesSeen - 1 - lag) % contextFrames).toInt() else 0
            val off = slot * baseDim
            for (j in 0 until baseDim) {
                acc += (ring[off + j] - mean[col]) * invScaleWeight[col]
                col++
            }
        }
        rawScore = sigmoid(acc)

        if (recentCount < smoothFrames) {
            recentCount++
        } else {
            recentSum -= recent[recentSlot]
        }
        recent[recentSlot] = rawScore
        recentSum += rawScore
        recentSlot = (recentSlot + 1) % smoothFrames
        smoothedScore = (recentSum / recentCount).toFloat()

        shouldMute =
            if (shouldMute) smoothedScore >= offThreshold else smoothedScore >= onThreshold
        return shouldMute
    }

    private fun sigmoid(x: Double): Float {
        val c = if (x > 30.0) 30.0 else if (x < -30.0) -30.0 else x
        return (1.0 / (1.0 + Math.exp(-c))).toFloat()
    }
}
