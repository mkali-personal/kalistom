package com.adsfilter

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.view.View
import android.view.WindowInsets
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale

/**
 * Phase 1 UI: start/stop the recorder, drop an alignment marker, and see what has been captured.
 * Deliberately minimal - the app's job in this phase is to record reliably, not to look nice.
 */
class MainActivity : Activity() {

    private lateinit var logView: TextView
    private lateinit var scroll: ScrollView

    private val projectionRequest = 1001
    private val permissionRequest = 1002

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }
        // targetSdk 35 forces edge-to-edge; without this the top row hides behind the system bars.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            root.setOnApplyWindowInsetsListener { v, insets ->
                val b = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.ime())
                v.setPadding(pad + b.left, pad + b.top, pad + b.right, pad + b.bottom)
                insets
            }
        } else {
            root.fitsSystemWindows = true
        }

        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        row.addView(button("Start recording") { ensurePermissionsThenStart() }, equal())
        row.addView(button("Stop") { send(RecorderService.ACTION_STOP) }, equal())
        root.addView(row)

        val row2 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        row2.addView(button("Mark") { send(RecorderService.ACTION_MARK) }, equal())
        row2.addView(button("Sessions") { listSessions() }, equal())
        root.addView(row2)

        root.addView(button("Attenuation vs capture test (75s)") {
            send(RecorderService.ACTION_ATTEN_TEST)
        })

        logView = TextView(this).apply {
            textSize = 11f
            typeface = android.graphics.Typeface.MONOSPACE
            setTextIsSelectable(true)
        }
        scroll = ScrollView(this).apply {
            addView(logView)
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f)
        }
        root.addView(scroll)

        root.addView(TextView(this).apply {
            text = "Recording is gated on playback: a session opens when audio starts and " +
                "closes 30 s after it stops, so every session is contiguous."
            textSize = 11f
        })

        setContentView(root)

        RecorderService.listener = { line -> runOnUiThread { append(line) } }
        synchronized(RecorderService.logLines) { RecorderService.logLines.forEach { append(it) } }
    }

    override fun onDestroy() {
        RecorderService.listener = null
        super.onDestroy()
    }

    private fun button(label: String, onClick: () -> Unit) = Button(this).apply {
        text = label
        setOnClickListener { onClick() }
    }

    private fun equal() =
        LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)

    private fun append(line: String) {
        logView.append(line + "\n")
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun send(action: String) {
        startService(Intent(this, RecorderService::class.java).setAction(action))
    }

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
    }

    private fun listSessions() {
        val dir = File(getExternalFilesDir(null), "sessions")
        val wavs = dir.listFiles { f -> f.name.endsWith(".wav") }?.sortedBy { it.name }
        if (wavs.isNullOrEmpty()) {
            append("(no sessions in ${dir.absolutePath})")
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
