package com.adsfilter.spike

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File

/**
 * Phase 0b spike UI. Flow:
 *   1. start audio playing in the app under test
 *   2. type its name in the label box
 *   3. Start -> grant the projection prompt -> let it run ~20 s -> Stop
 *   4. read the VERDICT line
 */
class MainActivity : Activity() {

    private lateinit var logView: TextView
    private lateinit var scroll: ScrollView
    private lateinit var labelField: EditText

    private val projectionRequest = 1001
    private val permissionRequest = 1002

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }

        // targetSdk 35 forces edge-to-edge: without this the top rows render behind the
        // status/action bar and the bottom hint behind the nav bar.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            root.setOnApplyWindowInsetsListener { v, insets ->
                val bars = insets.getInsets(
                    WindowInsets.Type.systemBars() or WindowInsets.Type.ime()
                )
                v.setPadding(pad + bars.left, pad + bars.top, pad + bars.right, pad + bars.bottom)
                insets
            }
        } else {
            root.fitsSystemWindows = true
        }

        labelField = EditText(this).apply {
            hint = "app under test (e.g. spotify, youtube, antennapod)"
            inputType = InputType.TYPE_CLASS_TEXT
            setSingleLine()
        }
        root.addView(labelField)

        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        row.addView(Button(this).apply {
            text = "Start capture"
            setOnClickListener { ensurePermissionsThenStart() }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row.addView(Button(this).apply {
            text = "Stop"
            setOnClickListener { stopCapture() }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(row)

        val row2 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        row2.addView(Button(this).apply {
            text = "List files"
            setOnClickListener { listFiles() }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row2.addView(Button(this).apply {
            text = "Clear log"
            setOnClickListener {
                synchronized(CaptureService.logLines) { CaptureService.logLines.clear() }
                logView.text = ""
            }
        }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(row2)

        root.addView(Button(this).apply {
            text = "Test global attenuation (session 0)"
            setOnClickListener { runEffectProbe() }
        })

        logView = TextView(this).apply {
            textSize = 11f
            typeface = android.graphics.Typeface.MONOSPACE
            setTextIsSelectable(true)
        }
        scroll = ScrollView(this).apply {
            addView(logView)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }
        root.addView(scroll)

        val hint = TextView(this).apply {
            text = "Start playback in the target app FIRST, then Start capture. " +
                "Let it run ~20 s, then Stop and read the VERDICT."
            textSize = 11f
            gravity = Gravity.CENTER
        }
        root.addView(hint)

        setContentView(root)

        CaptureService.listener = { line -> runOnUiThread { append(line) } }
        synchronized(CaptureService.logLines) {
            CaptureService.logLines.forEach { append(it) }
        }
    }

    override fun onDestroy() {
        CaptureService.listener = null
        super.onDestroy()
    }

    private fun append(line: String) {
        logView.append(line + "\n")
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun ensurePermissionsThenStart() {
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
        if (requestCode == permissionRequest) {
            val denied = permissions.indices.filter { grantResults[it] != PackageManager.PERMISSION_GRANTED }
            if (denied.isEmpty()) {
                requestProjection()
            } else {
                append("ERROR: permission denied: " + denied.joinToString { permissions[it] })
            }
        }
    }

    private fun requestProjection() {
        val mpm = getSystemService(MediaProjectionManager::class.java)
        // Token is single-use on Android 14+, so this prompt appears on every start.
        startActivityForResult(mpm.createScreenCaptureIntent(), projectionRequest)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != projectionRequest) return
        if (resultCode != RESULT_OK || data == null) {
            append("ERROR: projection consent denied (resultCode=" + resultCode + ")")
            return
        }
        val label = labelField.text.toString().trim()
            .replace(Regex("[^A-Za-z0-9_.-]"), "_")
            .ifBlank { "unknown" }

        val svc = Intent(this, CaptureService::class.java).apply {
            putExtra(CaptureService.EXTRA_RESULT_CODE, resultCode)
            putExtra(CaptureService.EXTRA_DATA, data)
            putExtra(CaptureService.EXTRA_LABEL, label)
        }
        startForegroundService(svc)
    }

    private fun stopCapture() {
        val svc = Intent(this, CaptureService::class.java).apply {
            action = CaptureService.ACTION_STOP
        }
        startService(svc)
    }

    /**
     * Attaches attenuating effects to the global output mix, holding each for 4 s so the
     * drop (or absence of one) is audible. Needs audio already playing.
     */
    private fun runEffectProbe() {
        Thread({
            EffectProbe.run(
                log = { line -> CaptureService.log(line) },
                hold = { try { Thread.sleep(4000) } catch (_: InterruptedException) {} }
            )
        }, "effect-probe").start()
    }

    private fun listFiles() {
        val dir = File(getExternalFilesDir(null), "spike")
        val files = dir.listFiles()
        if (files == null || files.isEmpty()) {
            append("(no files in " + dir.absolutePath + ")")
            return
        }
        append("--- " + dir.absolutePath + " ---")
        files.sortedBy { it.name }.forEach {
            append(String.format("%-44s %6d KB", it.name, it.length() / 1024))
        }
    }
}
