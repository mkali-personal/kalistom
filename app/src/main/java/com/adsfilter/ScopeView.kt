package com.adsfilter

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Live view of where the audio sits in the decision, and where it has just been.
 *
 * x is the model's smoothed score, so the shaded bands are exactly the mute rule - see [Scope] for
 * why this rather than PCA. y is YAMNet's "Music" score, which puts the model's known weakness on
 * screen: a dot travelling right while high up is music being mistaken for advertising.
 *
 * The x axis is warped to log-odds. On a linear probability axis the thresholds 0.70 and 0.99 sit
 * almost on top of each other and the mute band is a sliver, so all the interesting movement
 * happens in the last two percent of the width. In log-odds the same interval is legible and the
 * dot's approach to the threshold is visible rather than sudden.
 */
class ScopeView(context: Context) : View(context) {

    private val p = Palette(context)

    private val bandClear = Paint().apply { isAntiAlias = true }
    private val bandHyst = Paint().apply { isAntiAlias = true }
    private val bandMute = Paint().apply { isAntiAlias = true }
    private val gridPaint = Paint().apply {
        isAntiAlias = true; strokeWidth = 1f; style = Paint.Style.STROKE
    }
    private val textPaint = Paint().apply { isAntiAlias = true; textSize = 22f }
    private val trailPaint = Paint().apply { isAntiAlias = true; style = Paint.Style.FILL }
    private val dotPaint = Paint().apply { isAntiAlias = true; style = Paint.Style.FILL }
    private val ringPaint = Paint().apply {
        isAntiAlias = true; style = Paint.Style.STROKE; strokeWidth = 3f
    }

    private val sScore = FloatArray(Scope.CAPACITY)
    private val sMusic = FloatArray(Scope.CAPACITY)
    private val sMuted = BooleanArray(Scope.CAPACITY)

    private val plot = RectF()

    init {
        val dark = (context.resources.configuration.uiMode and
            android.content.res.Configuration.UI_MODE_NIGHT_MASK) ==
            android.content.res.Configuration.UI_MODE_NIGHT_YES
        bandClear.color = if (dark) Color.parseColor("#151A22") else Color.parseColor("#F2F5FA")
        bandHyst.color = if (dark) Color.parseColor("#2A2418") else Color.parseColor("#FBF3E0")
        bandMute.color = if (dark) Color.parseColor("#2A1A1C") else Color.parseColor("#FDECEC")
        gridPaint.color = if (dark) Color.parseColor("#2C323C") else Color.parseColor("#D7DCE5")
        textPaint.color = p.muted
    }

    /** Probability -> log-odds, clipped. The clip stops a saturated 1.0 flying off the canvas. */
    private fun warp(v: Float): Float {
        val q = min(max(v.toDouble(), 1e-4), 1.0 - 1e-4)
        return (ln(q / (1.0 - q))).toFloat().coerceIn(-LIMIT, LIMIT)
    }

    private fun xOf(score: Float) = plot.left + (warp(score) + LIMIT) / (2 * LIMIT) * plot.width()
    private fun yOf(music: Float) = plot.bottom - music.coerceIn(0f, 1f) * plot.height()

    override fun onDraw(canvas: Canvas) {
        val padL = 8f
        val padR = 8f
        val padT = 8f
        val padB = 34f
        plot.set(padL, padT, width - padR, height - padB)

        val onX = xOf(Scope.onThreshold)
        val offX = xOf(Scope.offThreshold)

        // Bands: released, hysteresis corridor, muting. Correct by construction, because x IS the
        // quantity the thresholds are compared against.
        canvas.drawRect(plot.left, plot.top, offX, plot.bottom, bandClear)
        canvas.drawRect(offX, plot.top, onX, plot.bottom, bandHyst)
        canvas.drawRect(onX, plot.top, plot.right, plot.bottom, bandMute)

        for (t in TICKS) {
            val x = xOf(t)
            canvas.drawLine(x, plot.top, x, plot.bottom, gridPaint)
            val lbl = if (t >= 0.99f) "0.99" else if (t <= 0.01f) "0.01" else t.toString()
            canvas.drawText(lbl, x - textPaint.measureText(lbl) / 2, height - 12f, textPaint)
        }
        canvas.drawRect(plot, gridPaint)

        canvas.drawText("music", plot.left + 8f, plot.top + 24f, textPaint)
        val axis = "ad score"
        canvas.drawText(axis, plot.right - textPaint.measureText(axis) - 8f,
            plot.bottom - 10f, textPaint)

        if (!Scope.active) {
            val msg = "waiting for playback"
            canvas.drawText(msg, plot.centerX() - textPaint.measureText(msg) / 2,
                plot.centerY(), textPaint)
            return
        }

        val n = Scope.snapshot(sScore, sMusic, sMuted)
        if (n == 0) return

        // Oldest faintest, so the direction of travel is readable without arrows.
        for (k in 0 until n - 1) {
            val a = (30 + 150 * k / max(n - 1, 1))
            trailPaint.color = Color.argb(a, Color.red(p.accent), Color.green(p.accent),
                Color.blue(p.accent))
            val r = 2f + 3f * k / max(n - 1, 1)
            canvas.drawCircle(xOf(sScore[k]), yOf(sMusic[k]), r, trailPaint)
        }

        val i = n - 1
        val muted = sMuted[i]
        dotPaint.color = if (muted) p.live else p.accent
        ringPaint.color = dotPaint.color
        val cx = xOf(sScore[i])
        val cy = yOf(sMusic[i])
        canvas.drawCircle(cx, cy, 9f, dotPaint)
        canvas.drawCircle(cx, cy, 15f, ringPaint)

        val read = String.format("p=%.3f  music=%.2f%s", sScore[i], sMusic[i],
            if (muted) "  MUTED" else "")
        canvas.drawText(read, plot.left + 8f, plot.bottom - 10f, textPaint)
    }

    companion object {
        /** Log-odds range shown. +-7 covers p from about 0.001 to 0.999. */
        private const val LIMIT = 7f
        private val TICKS = floatArrayOf(0.01f, 0.1f, 0.5f, 0.9f, 0.99f)
    }
}
