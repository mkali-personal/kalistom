package com.adsfilter

import android.content.Context
import android.util.Log
import org.json.JSONObject

/**
 * The trained head, running on the phone: embedding stream in, mute decision out.
 *
 * Deliberately the dumbest part of the system. YAMNet does the listening and the desktop does the
 * learning; this is a small head over the last few frames, an optional running average and two
 * thresholds. Everything it needs arrives in head_weights.json, which the trainer writes -
 * including which head it is and the thresholds, so neither retraining nor retuning means editing
 * Kotlin.
 *
 * TWO HEADS, chosen by the file's "type":
 *
 *  - "logistic_regression": one dot product over the stack of the last `context_frames` base
 *    frames (trainer/context_lr.py).
 *  - "mlp": a small hidden layer applied to each frame separately, then linear over the last
 *    `context_frames` of those (trainer/context_mlp.py). Each frame's hidden vector is computed
 *    once, when the frame arrives, and kept in a ring - so a new frame costs one 1025 x hidden
 *    product plus a context_frames x hidden dot product, never a recomputation of the history.
 *
 * WHY THE ORDER OF OPERATIONS MATTERS. This has to reproduce the trainer exactly, because a
 * mismatch here does not crash, it silently degrades: the model would be scoring vectors assembled
 * differently from the ones it was fitted on. Three places where it is easy to get wrong, all of
 * them checked by trainer/parity_check.py against a real session:
 *
 *  1. Column order. The base frame is 1024 embedding values followed by ONE loudness value,
 *     scaled (dBFS + 60) / 60 - the same expression as in train.py.
 *  2. Tap order. The context is oldest frame first: [frame i-9, i-8, ... , i]. That is
 *     `stack_context`'s `arange(n)[::-1]`, and reversing it would still produce a plausible
 *     number from the wrong weights.
 *  3. Edges. Training clamped at each recording's start, repeating the first frame rather than
 *     dropping rows. A session here is a recording, so the first frames repeat frame 0 too - for
 *     the mlp, frame 0's hidden vector.
 *
 * SMOOTHING. The file's `smoothing.frames` sets a running average of the score before the
 * thresholds; 1 means none, and `on`/`off` then apply to each frame's own score. The logistic
 * head shipped with an 11-frame average because its single-frame score was too jumpy; the mlp's
 * score is clean enough that plain hysteresis - mute above 0.999, release below 0.2 - did as well
 * on held-out audio and entered breaks sooner. Because the thresholds mean different things with
 * and without the average, they always travel in the same file as the smoothing block.
 */
class Detector private constructor(
    private val scorer: Scorer,
    private val smoothFrames: Int,
    val onThreshold: Float,
    val offThreshold: Float,
) {

    /** One head: takes each new base frame, returns the logit for the latest frame. */
    private interface Scorer {
        fun push(frame: FloatArray): Double
        fun reset()
    }

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
            val type = o.optString("type", "logistic_regression")
            val scorer: Scorer = when (type) {
                "logistic_regression" -> Linear(
                    ctx, base,
                    mean = floats(o, "mean", inputDim),
                    scale = floats(o, "scale", inputDim),
                    weights = floats(o, "weights", inputDim),
                    bias = o.getDouble("bias").toFloat(),
                )
                "mlp" -> {
                    val hidden = o.getInt("hidden")
                    Mlp(
                        ctx, base, hidden,
                        mean = floats(o, "mean", base),
                        scale = floats(o, "scale", base),
                        w1 = floats(o, "w1", hidden * base),
                        b1 = floats(o, "b1", hidden),
                        w2 = floats(o, "w2", ctx * hidden),
                        b2 = o.getDouble("b2").toFloat(),
                    )
                }
                else -> throw IllegalArgumentException("unknown head type '$type'")
            }
            val smoothing = o.optJSONObject("smoothing")
            val d = Detector(
                scorer = scorer,
                smoothFrames = maxOf(1, smoothing?.optInt("frames", 1) ?: 1),
                onThreshold = o.getDouble("on").toFloat(),
                offThreshold = o.getDouble("off").toFloat(),
            )
            Log.i(
                TAG,
                "loaded $type: $ctx taps x $base = $inputDim inputs, smoothing ${d.smoothFrames} frames " +
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

        /**
         * Which ring slot holds the frame `lag` frames back, clamped to frame 0 at a session's
         * start. Frame k lives in slot k % taps; clamping only happens while framesSeen <= lag <
         * taps, so frame 0 is still in slot 0 and unoverwritten. Shared by both heads, and
         * transliterated line for line in trainer/test_detector_logic.py.
         */
        private fun slotFor(framesSeen: Long, lag: Int, taps: Int): Int =
            if (framesSeen > lag) ((framesSeen - 1 - lag) % taps).toInt() else 0
    }

    /** The logistic regression: the ring holds raw base frames; the dot product runs over all taps. */
    private class Linear(
        private val taps: Int,
        private val baseDim: Int,
        private val mean: FloatArray,
        scale: FloatArray,
        weights: FloatArray,
        private val bias: Float,
    ) : Scorer {
        /**
         * weight / scale, folded together once at load. The score needs
         * sum over columns of ((x - mean) / scale) * weight, and scale and weight are both fixed,
         * so the division is a constant - doing it per frame would be 10,250 divisions every
         * 0.48 s for no reason.
         */
        private val invScaleWeight = FloatArray(weights.size) { weights[it] / scale[it] }
        private val ring = FloatArray(taps * baseDim)
        private var framesSeen = 0L
        private var writeSlot = 0

        override fun push(frame: FloatArray): Double {
            System.arraycopy(frame, 0, ring, writeSlot * baseDim, baseDim)
            writeSlot = (writeSlot + 1) % taps
            framesSeen++
            var acc = bias.toDouble()
            var col = 0
            for (tap in 0 until taps) {
                val off = slotFor(framesSeen, taps - 1 - tap, taps) * baseDim
                for (j in 0 until baseDim) {
                    acc += (ring[off + j] - mean[col]) * invScaleWeight[col]
                    col++
                }
            }
            return acc
        }

        override fun reset() {
            java.util.Arrays.fill(ring, 0f)
            framesSeen = 0
            writeSlot = 0
        }
    }

    /**
     * The mlp: each frame becomes `hidden` numbers once, h = relu(W1 . (x - mean) / scale + b1),
     * and the ring holds those; the score is b2 + sum over taps of W2[tap] . h(frame for that tap).
     */
    private class Mlp(
        private val taps: Int,
        private val baseDim: Int,
        private val hidden: Int,
        mean: FloatArray,
        scale: FloatArray,
        w1: FloatArray,
        b1: FloatArray,
        private val w2: FloatArray,
        private val b2: Float,
    ) : Scorer {
        /**
         * Standardisation folded into the first layer at load: W1 . ((x - mean) / scale) + b1 is
         * (W1 / scale) . x + (b1 - (W1 / scale) . mean). Both parts are constants, so a frame
         * costs a plain matrix-vector product and no per-column subtraction or division.
         */
        private val w1s = FloatArray(hidden * baseDim) { w1[it] / scale[it % baseDim] }
        private val c1 = DoubleArray(hidden) { j ->
            var s = b1[j].toDouble()
            for (i in 0 until baseDim) s -= w1s[j * baseDim + i].toDouble() * mean[i]
            s
        }
        private val ring = FloatArray(taps * hidden)
        private var framesSeen = 0L
        private var writeSlot = 0

        override fun push(frame: FloatArray): Double {
            val at = writeSlot * hidden
            for (j in 0 until hidden) {
                var s = c1[j]
                val row = j * baseDim
                for (i in 0 until baseDim) s += w1s[row + i] * frame[i]
                ring[at + j] = if (s > 0.0) s.toFloat() else 0f
            }
            writeSlot = (writeSlot + 1) % taps
            framesSeen++
            var acc = b2.toDouble()
            for (tap in 0 until taps) {
                val off = slotFor(framesSeen, taps - 1 - tap, taps) * hidden
                val w = tap * hidden
                for (j in 0 until hidden) acc += w2[w + j] * ring[off + j]
            }
            return acc
        }

        override fun reset() {
            java.util.Arrays.fill(ring, 0f)
            framesSeen = 0
            writeSlot = 0
        }
    }

    /** One base frame, assembled here so both heads see the same columns: embedding, then loudness. */
    private val frame = FloatArray(Yamnet.EMBEDDING_DIM + 1)

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
        scorer.reset()
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
     * nothing: about 10k multiply-adds for the logistic head and 17k for a 16-unit mlp per 0.48 s,
     * which is noise next to YAMNet's roughly 69 million.
     */
    fun push(embedding: FloatArray, rmsDbfs: Float): Boolean {
        System.arraycopy(embedding, 0, frame, 0, Yamnet.EMBEDDING_DIM)
        frame[Yamnet.EMBEDDING_DIM] = (rmsDbfs + 60f) / 60f
        rawScore = sigmoid(scorer.push(frame))

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
