package com.adsfilter

import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Turns a stream of PCM samples into YAMNet frames, guaranteeing that frame k covers samples
 * [k*HOP, k*HOP + WINDOW) counting from the first sample pushed.
 *
 * Live recording and offline processing of desktop files both go through this one class, so the
 * two cannot drift apart. That matters more than it looks: the whole point of processing desktop
 * audio on the phone is that it goes through the exact code path production uses, and a second
 * copy of the framing logic would quietly defeat that.
 *
 * WINDOW is not a multiple of HOP, so the buffer must hold more than one window and advance by
 * exactly HOP per frame. An earlier shift-then-append version dropped most of each push from the
 * analysis window while the audio still reached the file, desynchronising embeddings from audio
 * with no visible symptom.
 */
class FrameEmitter(
    private val net: Yamnet,
    private val onFrame: (index: Long, embedding: FloatArray, rmsDbfs: Float,
                          topIdx: IntArray, topScore: FloatArray) -> Unit
) {
    private val buf = FloatArray(Yamnet.WINDOW + 2 * Yamnet.HOP)
    private val window = FloatArray(Yamnet.WINDOW)
    private var fill = 0
    private var frameIndex = 0L

    /** Frames emitted so far. */
    fun frames(): Long = frameIndex

    /** Discards buffered audio and restarts numbering - call when starting a new session. */
    fun reset() {
        fill = 0
        frameIndex = 0
    }

    /** Feeds [n] samples of [pcm]; emits every frame they complete. */
    fun push(pcm: ShortArray, n: Int) {
        var off = 0
        while (off < n) {
            val room = minOf(n - off, buf.size - fill)
            for (i in 0 until room) {
                buf[fill + i] = pcm[off + i] / 32768.0f
            }
            fill += room
            off += room
            drain()
            // If the buffer filled without completing a frame something is badly wrong, but
            // guard anyway so a bad input cannot spin forever.
            if (room == 0) break
        }
    }

    private fun drain() {
        while (fill >= Yamnet.WINDOW) {
            System.arraycopy(buf, 0, window, 0, Yamnet.WINDOW)
            // rms describes this frame's own window, which is what lets the desktop verifier
            // recompute it from the audio and prove the two are aligned.
            val rms = rmsDbfs(window)
            val r = net.run(window)
            val top = topK(r.scores, 3)
            onFrame(frameIndex, r.embedding, rms, top.first, top.second)
            frameIndex++
            System.arraycopy(buf, Yamnet.HOP, buf, 0, fill - Yamnet.HOP)
            fill -= Yamnet.HOP
        }
    }

    companion object {
        fun rmsDbfs(w: FloatArray): Float {
            var sum = 0.0
            for (v in w) sum += v.toDouble() * v.toDouble()
            val rms = sqrt(sum / w.size)
            return if (rms <= 0.0) -120f
            else (20.0 * log10(rms)).coerceAtLeast(-120.0).toFloat()
        }

        fun topK(scores: FloatArray, k: Int): Pair<IntArray, FloatArray> {
            val idx = IntArray(k)
            val sc = FloatArray(k)
            for (r in 0 until k) {
                var best = -1
                var bestV = Float.NEGATIVE_INFINITY
                for (i in scores.indices) {
                    if (scores[i] > bestV && (0 until r).none { idx[it] == i }) {
                        best = i; bestV = scores[i]
                    }
                }
                idx[r] = best.coerceAtLeast(0)
                sc[r] = if (best >= 0) scores[best] else 0f
            }
            return idx to sc
        }
    }
}
