package com.adsfilter

/**
 * The last half-minute of the detector's trajectory, for [ScopeView] to draw.
 *
 * WHAT THE TWO AXES ARE, AND WHY NOT PCA. The obvious visualisation is PCA of the embedding space
 * with the classified regions shaded. Measured on this data, that picture would mislead:
 *
 *   - Two principal components capture 7.4 % of the variance, and a frame's score recomputed from
 *     its 2-D reconstruction agrees with its real score for only 88 % of frames. One dot in eight
 *     would sit on the wrong colour.
 *   - The direction the classifier actually uses carries 0.03 % of the variance. PCA is
 *     unsupervised: it finds where the audio varies most, which is speech against music, loudness
 *     and timbre. The advertising/content distinction is nearly orthogonal to all of it.
 *   - The head is linear, so the decision boundary in ANY 2-D linear projection is exactly a
 *     straight line. Sampling a 100x100 grid would spend ten thousand evaluations rediscovering
 *     that, and the closed form here is two multiplications.
 *
 * So the horizontal axis is the model's own smoothed score, which makes the shaded bands correct
 * by construction rather than approximately: left of `off` is release, right of `on` is mute, and
 * the corridor between them is the hysteresis. The vertical axis is YAMNet's "Music" score, chosen
 * because the model's known weakness is confusing music with advertising - a dot drifting right
 * while riding high is that failure, visible as it happens.
 *
 * Written from the capture thread and read from the UI thread. The ring is plain arrays with a
 * volatile cursor: a torn read costs one stale dot in one frame of animation, which is not worth a
 * lock on the audio path.
 */
object Scope {

    /** 0.48 s per frame, so this is a little over half a minute of trail. */
    const val CAPACITY = 64

    private val scoreBuf = FloatArray(CAPACITY)
    private val musicBuf = FloatArray(CAPACITY)
    private val mutedBuf = BooleanArray(CAPACITY)

    @Volatile private var writeAt = 0
    @Volatile private var count = 0

    /** Thresholds the bands are drawn at; the service sets these from head_weights.json. */
    @Volatile var onThreshold = 0.99f
    @Volatile var offThreshold = 0.70f
    @Volatile var active = false

    fun push(smoothedScore: Float, music: Float, muted: Boolean) {
        val i = writeAt
        scoreBuf[i] = smoothedScore
        musicBuf[i] = music
        mutedBuf[i] = muted
        writeAt = (i + 1) % CAPACITY
        if (count < CAPACITY) count++
        active = true
    }

    fun clear() {
        count = 0
        writeAt = 0
        active = false
    }

    /**
     * Copies the trail oldest-first into the caller's arrays, returning how many points were
     * written. Copying keeps the UI off the live buffers entirely.
     */
    fun snapshot(score: FloatArray, music: FloatArray, muted: BooleanArray): Int {
        val n = count
        val end = writeAt
        for (k in 0 until n) {
            val i = ((end - n + k) % CAPACITY + CAPACITY) % CAPACITY
            score[k] = scoreBuf[i]
            music[k] = musicBuf[i]
            muted[k] = mutedBuf[i]
        }
        return n
    }
}
