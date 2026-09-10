package com.adsfilter

import android.content.Context
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.content.res.ColorStateList
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView

/**
 * A small palette and a few view builders, so the screen can be laid out in code without every
 * measurement and colour being spelled out at the call site.
 *
 * There is no XML here on purpose. The app is one screen and will lose most of its controls once
 * the detector runs by itself; a layout resource, a styles file and a colour file would be three
 * more places to keep in step for something that is going to shrink.
 */
class Palette(ctx: Context) {
    private val dark = (ctx.resources.configuration.uiMode and
        Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES

    val bg = if (dark) Color.parseColor("#0F1116") else Color.parseColor("#F6F7F9")
    val surface = if (dark) Color.parseColor("#181B22") else Color.WHITE
    val border = if (dark) Color.parseColor("#262B35") else Color.parseColor("#E3E6EB")
    val text = if (dark) Color.parseColor("#E8EAEE") else Color.parseColor("#171A20")
    val muted = if (dark) Color.parseColor("#878E9C") else Color.parseColor("#697080")
    val accent = if (dark) Color.parseColor("#5B8DEF") else Color.parseColor("#2F6BE0")
    val onAccent = Color.WHITE
    val live = Color.parseColor("#E5484D")
    val idle = if (dark) Color.parseColor("#3A4150") else Color.parseColor("#C6CBD4")
}

fun Context.dp(v: Float): Int =
    TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics).toInt()

/** Rounded rectangle, optionally outlined - the only shape this screen uses. */
fun roundedRect(fill: Int, radiusPx: Int, strokePx: Int = 0, strokeColor: Int = 0) =
    GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = radiusPx.toFloat()
        setColor(fill)
        if (strokePx > 0) setStroke(strokePx, strokeColor)
    }

/** The same shape with a touch ripple over it, so buttons feel like buttons. */
fun rippled(fill: Int, radiusPx: Int, rippleColor: Int, strokePx: Int = 0, strokeColor: Int = 0) =
    RippleDrawable(
        ColorStateList.valueOf(rippleColor),
        roundedRect(fill, radiusPx, strokePx, strokeColor),
        null
    )

fun Context.label(txt: String, sizeSp: Float, colour: Int, bold: Boolean = false) =
    TextView(this).apply {
        text = txt
        textSize = sizeSp
        setTextColor(colour)
        if (bold) typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

fun Context.row(vararg children: View, gap: Float = 10f): LinearLayout =
    LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        children.forEachIndexed { i, child ->
            val lp = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            if (i > 0) lp.leftMargin = dp(gap)
            addView(child, lp)
        }
    }
