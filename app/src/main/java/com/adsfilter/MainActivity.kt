package com.adsfilter

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale

/**
 * The one screen: start and stop the recorder, drop an alignment marker, and see what has been
 * captured.
 *
 * Most of these controls exist for the current phase and will go once the detector runs by itself,
 * so the layout is arranged around that: one primary action that fills the eye, the day-to-day
 * controls beside it, and everything that is really a developer tool pushed down into its own
 * quiet group rather than sitting on the same footing as the button you press every day.
 */
class MainActivity : Activity() {

    private lateinit var p: Palette
    private lateinit var logView: TextView
    private lateinit var scroll: ScrollView
    private lateinit var primary: Button
    private lateinit var statusDot: View
    private lateinit var statusText: TextView
    private lateinit var statusDetail: TextView

    private val projectionRequest = 1001
    private val permissionRequest = 1002

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        p = Palette(this)

        val pad = dp(20f)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(p.bg)
            setPadding(pad, pad, pad, pad)
        }
        // targetSdk 35 forces edge-to-edge; without this the header hides behind the status bar.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            root.setOnApplyWindowInsetsListener { v, insets ->
                val b = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.ime())
                v.setPadding(pad + b.left, pad + b.top, pad + b.right, pad + b.bottom)
                insets
            }
        } else {
            root.fitsSystemWindows = true
        }

        root.addView(label("Kalistom", 26f, p.text, bold = true))
        // The margin has to be given to addView: a view's layoutParams is null until a parent
        // assigns it, so setting it inside apply{} on a freshly built view does nothing at all.
        root.addView(label("Kan Reshet Bet · playback capture", 13f, p.muted), marginTop(2f))

        root.addView(statusCard(), marginTop(18f))
        root.addView(primaryButton(), marginTop(14f))
        root.addView(
            row(
                secondary("Mark") { send(RecorderService.ACTION_MARK) },
                secondary("Sessions") { listSessions() }
            ),
            marginTop(10f)
        )

        root.addView(label("TOOLS", 11f, p.muted, bold = true).apply {
            letterSpacing = 0.12f
        }, marginTop(22f))
        root.addView(quiet("Attenuation vs capture test") {
            send(RecorderService.ACTION_ATTEN_TEST)
        }, marginTop(6f))
        root.addView(quiet("Process desktop recordings") {
            send(RecorderService.ACTION_PROCESS_INBOX)
        }, marginTop(6f))

        root.addView(label("ACTIVITY", 11f, p.muted, bold = true).apply {
            letterSpacing = 0.12f
        }, marginTop(22f))
        root.addView(activityPanel(), LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
        ).apply { topMargin = dp(6f) })

        root.addView(label(
            "A session opens when audio starts playing and closes 30 seconds after it stops, " +
                "so every session is one continuous stretch. The phone's volume does not matter.",
            11f, p.muted
        ), marginTop(12f))

        setContentView(root)

        RecorderService.listener = { line ->
            runOnUiThread { append(line); refreshStatus() }
        }
        synchronized(RecorderService.logLines) { RecorderService.logLines.forEach { append(it) } }
        refreshStatus()
    }

    override fun onResume() {
        super.onResume()
        refreshStatus()
    }

    override fun onDestroy() {
        RecorderService.listener = null
        super.onDestroy()
    }

    // ---------------------------------------------------------------- pieces

    /** Whether the recorder is running, said plainly and visible from across the room. */
    private fun statusCard(): View {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = roundedRect(p.surface, dp(16f), dp(1f), p.border)
            setPadding(dp(16f), dp(16f), dp(16f), dp(16f))
        }
        statusDot = View(this).apply {
            background = roundedRect(p.idle, dp(5f))
        }
        card.addView(statusDot, LinearLayout.LayoutParams(dp(10f), dp(10f)))

        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        statusText = label("Idle", 16f, p.text, bold = true)
        statusDetail = label("Not recording", 12f, p.muted)
        col.addView(statusText)
        col.addView(statusDetail)
        card.addView(col, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { leftMargin = dp(12f) })
        return card
    }

    private fun primaryButton(): Button {
        primary = Button(this).apply {
            text = "Start recording"
            isAllCaps = false
            textSize = 16f
            setTextColor(p.onAccent)
            typeface = Typeface.DEFAULT_BOLD
            stateListAnimator = null
            background = rippled(p.accent, dp(14f), p.onAccent and 0x40FFFFFF)
            setPadding(0, dp(16f), 0, dp(16f))
            setOnClickListener {
                if (RecorderService.isRunning) send(RecorderService.ACTION_STOP)
                else ensurePermissionsThenStart()
            }
        }
        return primary
    }

    /** Outlined, for the things you press while a session is running. */
    private fun secondary(txt: String, onClick: () -> Unit) = Button(this).apply {
        text = txt
        isAllCaps = false
        textSize = 15f
        setTextColor(p.text)
        stateListAnimator = null
        background = rippled(p.surface, dp(12f), p.accent and 0x33FFFFFF, dp(1f), p.border)
        setPadding(0, dp(12f), 0, dp(12f))
        setOnClickListener { onClick() }
    }

    /** Flat and left-aligned, so the developer tools read as a list rather than as buttons. */
    private fun quiet(txt: String, onClick: () -> Unit) = Button(this).apply {
        text = txt
        isAllCaps = false
        textSize = 14f
        setTextColor(p.muted)
        gravity = Gravity.CENTER_VERTICAL or Gravity.START
        stateListAnimator = null
        background = rippled(p.bg, dp(10f), p.accent and 0x22FFFFFF)
        setPadding(dp(12f), dp(11f), dp(12f), dp(11f))
        minHeight = 0
        minimumHeight = 0
        setOnClickListener { onClick() }
    }

    private fun activityPanel(): View {
        logView = TextView(this).apply {
            textSize = 11f
            typeface = Typeface.MONOSPACE
            setTextColor(p.muted)
            setTextIsSelectable(true)
            setLineSpacing(dp(2f).toFloat(), 1f)
        }
        scroll = ScrollView(this).apply {
            addView(logView)
            background = roundedRect(p.surface, dp(14f), dp(1f), p.border)
            setPadding(dp(14f), dp(12f), dp(14f), dp(12f))
            clipToOutline = true
        }
        return scroll
    }

    private fun marginTop(v: Float) = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
    ).apply { topMargin = dp(v) }

    // ---------------------------------------------------------------- state

    private fun refreshStatus() {
        val running = RecorderService.isRunning
        primary.text = if (running) "Stop recording" else "Start recording"
        primary.background = rippled(
            if (running) p.live else p.accent, dp(14f), p.onAccent and 0x40FFFFFF
        )
        statusDot.background = roundedRect(if (running) p.live else p.idle, dp(5f))
        statusText.text = if (running) "Recording" else "Idle"
        statusText.setTextColor(if (running) p.live else p.text)
        statusDetail.text = RecorderService.status.ifBlank {
            if (running) "Waiting for playback" else "Not recording"
        }
    }

    private fun append(line: String) {
        logView.append(line + "\n")
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun send(action: String) {
        startService(Intent(this, RecorderService::class.java).setAction(action))
    }

    // ---------------------------------------------------------------- plumbing

    private fun ensurePermissionsThenStart() {
        // Starting twice was already harmless - the service ignores it - but it still made the
        // user grant a projection consent that was then thrown away. Say so instead.
        if (RecorderService.isRunning) {
            append("already recording (${RecorderService.status}) - Stop first")
            return
        }
        val needed = ArrayList<String>()
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            needed.add(Manifest.permission.RECORD_AUDIO)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (needed.isNotEmpty()) {
            requestPermissions(needed.toTypedArray(), permissionRequest)
            return
        }
        requestProjection()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != permissionRequest) return
        val denied = permissions.indices.filter { grantResults[it] != PackageManager.PERMISSION_GRANTED }
        if (denied.isEmpty()) requestProjection()
        else append("ERROR: denied ${denied.joinToString { permissions[it] }}")
    }

    private fun requestProjection() {
        val mpm = getSystemService(MediaProjectionManager::class.java)
        startActivityForResult(mpm.createScreenCaptureIntent(), projectionRequest)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != projectionRequest) return
        if (resultCode != RESULT_OK || data == null) {
            append("ERROR: projection consent denied")
            return
        }
        startForegroundService(
            Intent(this, RecorderService::class.java)
                .putExtra(RecorderService.EXTRA_RESULT_CODE, resultCode)
                .putExtra(RecorderService.EXTRA_DATA, data)
        )
        refreshStatus()
    }

    private fun listSessions() {
        val dir = File(getExternalFilesDir(null), "sessions")
        val wavs = dir.listFiles { f -> f.name.endsWith(".wav") }?.sortedBy { it.name }
        if (wavs.isNullOrEmpty()) {
            append("no sessions yet")
            return
        }
        append("--- ${wavs.size} session(s) ---")
        var totalMb = 0.0
        wavs.forEach { w ->
            val base = w.name.removeSuffix(".wav")
            val emb = File(dir, "$base.f16")
            val secs = (w.length() - 44).coerceAtLeast(0) / 2.0 / Yamnet.SAMPLE_RATE
            val frames = emb.length() / (Yamnet.EMBEDDING_DIM * 2L)
            totalMb += w.length() / 1e6 + emb.length() / 1e6
            append(
                String.format(
                    Locale.US, "%s  %6.0fs  %5d frames  %5.1f MB",
                    base, secs, frames, (w.length() + emb.length()) / 1e6
                )
            )
        }
        append(String.format(Locale.US, "total %.1f MB", totalMb))
    }
}
