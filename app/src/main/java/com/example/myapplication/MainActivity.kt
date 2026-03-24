package com.example.myapplication

import android.content.Context
import android.content.pm.ActivityInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.ui.theme.MyApplicationTheme
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    // Sensor values
    private var accX by mutableStateOf(0f)
    private var accY by mutableStateOf(0f)
    private var accZ by mutableStateOf(0f)
    private var gyroX by mutableStateOf(0f)
    private var gyroY by mutableStateOf(0f)
    private var gyroZ by mutableStateOf(0f)

    // Recording state
    private var isRecording by mutableStateOf(false)
    private var selectedLabel by mutableStateOf("standing_still")
    private var sessionId by mutableStateOf("")
    private var recordingStartTime by mutableStateOf(0L)
    private var sampleCount by mutableStateOf(0)
    private var elapsedSeconds by mutableStateOf(0)
    private var lastSavedFile by mutableStateOf("")
    private var statusMessage by mutableStateOf("Ready")

    private var csvWriter: BufferedWriter? = null
    private var csvFile: File? = null

    // Fixed experiment metadata
    private val phoneHand = "left_hand"
    private val screenDirection = "screen_toward_body"

    private val labels = listOf(
        "standing_still", "walking", "running", "stairs_up", "stairs_down"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        enableEdgeToEdge()

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        setContent {
            MyApplicationTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    DataCollectionScreen(Modifier.padding(innerPadding))
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
        if (isRecording) stopRecording()
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accX = event.values[0]
                accY = event.values[1]
                accZ = event.values[2]
                // CSV 기록은 accelerometer 이벤트 기준으로 작성
                if (isRecording) writeCsvRow()
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroX = event.values[0]
                gyroY = event.values[1]
                gyroZ = event.values[2]
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun writeCsvRow() {
        val now = System.currentTimeMillis()
        val elapsed = now - recordingStartTime
        elapsedSeconds = (elapsed / 1000).toInt()
        csvWriter?.let { w ->
            w.write(
                String.format(
                    Locale.US,
                    "%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    now, elapsed, selectedLabel, sessionId,
                    accX, accY, accZ, gyroX, gyroY, gyroZ
                )
            )
            w.newLine()
            sampleCount++
        }
    }

    private fun startRecording() {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        sessionId = "${selectedLabel}_$timestamp"

        val dir = getExternalFilesDir(null) ?: filesDir
        csvFile = File(dir, "${sessionId}.csv")

        csvWriter = BufferedWriter(FileWriter(csvFile)).apply {
            write("# phone_hand=$phoneHand\n")
            write("# screen_direction=$screenDirection\n")
            write("# device_model=${Build.MODEL}\n")
            write("# recording_start=$timestamp\n")
            write("timestamp_ms,elapsed_ms,label,session_id,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
        }

        recordingStartTime = System.currentTimeMillis()
        sampleCount = 0
        elapsedSeconds = 0
        isRecording = true
        statusMessage = "Recording..."
    }

    private fun stopRecording() {
        isRecording = false
        try {
            csvWriter?.flush()
            csvWriter?.close()
        } catch (_: Exception) { }
        csvWriter = null
        lastSavedFile = csvFile?.absolutePath ?: ""
        statusMessage = "Saved: ${csvFile?.name} ($sampleCount samples)"
    }

    @Composable
    fun DataCollectionScreen(modifier: Modifier = Modifier) {
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                "Data Collection",
                style = MaterialTheme.typography.headlineMedium
            )

            // Label selector
            Text("Activity Label:", style = MaterialTheme.typography.titleSmall)
            var expanded by remember { mutableStateOf(false) }
            Box {
                OutlinedButton(
                    onClick = { if (!isRecording) expanded = true },
                    enabled = !isRecording,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(selectedLabel)
                    Spacer(Modifier.weight(1f))
                    Text("▼")
                }
                DropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false }
                ) {
                    labels.forEach { label ->
                        DropdownMenuItem(
                            text = { Text(label) },
                            onClick = {
                                selectedLabel = label
                                expanded = false
                            }
                        )
                    }
                }
            }

            HorizontalDivider()

            // Accelerometer
            Text("Accelerometer (m/s²)", style = MaterialTheme.typography.titleSmall)
            Text(
                String.format(Locale.US, "X: %.4f   Y: %.4f   Z: %.4f", accX, accY, accZ),
                fontSize = 14.sp
            )

            // Gyroscope
            Text("Gyroscope (rad/s)", style = MaterialTheme.typography.titleSmall)
            Text(
                String.format(Locale.US, "X: %.4f   Y: %.4f   Z: %.4f", gyroX, gyroY, gyroZ),
                fontSize = 14.sp
            )

            HorizontalDivider()

            // Recording status
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    if (isRecording) "● RECORDING" else "○ IDLE",
                    color = if (isRecording) Color.Red else Color.Gray,
                    fontSize = 16.sp
                )
                if (isRecording) {
                    Spacer(Modifier.width(16.dp))
                    Text("${elapsedSeconds}s | $sampleCount samples", fontSize = 14.sp)
                }
            }

            if (sessionId.isNotEmpty()) {
                Text("Session: $sessionId", fontSize = 12.sp, color = Color.Gray)
            }

            // Start / Stop
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Button(
                    onClick = { startRecording() },
                    enabled = !isRecording,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Start")
                }
                Button(
                    onClick = { stopRecording() },
                    enabled = isRecording,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Text("Stop")
                }
            }

            HorizontalDivider()

            Text(statusMessage, fontSize = 14.sp)
            if (lastSavedFile.isNotEmpty()) {
                Text("Path: $lastSavedFile", fontSize = 11.sp, color = Color.Gray)
            }
        }
    }
}