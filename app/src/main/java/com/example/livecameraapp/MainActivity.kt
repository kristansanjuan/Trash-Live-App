package com.example.livecameraapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.widget.TextView
import android.widget.Toast
import androidx.camera.view.PreviewView
import com.example.livecameraapp.ml.ModelUnquant // Ensure this matches the generated model class name
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var model: ModelUnquant
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var resultTextView: TextView
    private var imageSize = 224  // The size of input image the model expects

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultTextView = findViewById(R.id.result)

        // Load the model
        model = ModelUnquant.newInstance(this)

        // Camera permission check
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(findViewById<PreviewView>(R.id.previewView).surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(imageSize, imageSize))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { imageAnalysis ->
                    imageAnalysis.setAnalyzer(cameraExecutor, { imageProxy: ImageProxy ->
                        val bitmap = imageProxy.toBitmap()
                        runModel(bitmap)
                        imageProxy.close()
                    })
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

            } catch (exc: Exception) {
                Toast.makeText(this, "Error binding camera: ${exc.message}", Toast.LENGTH_SHORT).show()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun runModel(bitmap: Bitmap) {
        try {
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(imageSize * imageSize)
            bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            var pixelIndex = 0
            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val pixelValue = intValues[pixelIndex++]
                    byteBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255f)
                    byteBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255f)
                    byteBuffer.putFloat((pixelValue and 0xFF) / 255f)
                }
            }

            inputFeature0.loadBuffer(byteBuffer)

            // Run inference
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            val confidences = outputFeature0.floatArray

            // Find the max confidence index
            val maxPos = confidences.indices.maxByOrNull { confidences[it] } ?: -1
            val classes = arrayOf("Biodegradable", "Non-biodegradable", "Recyclable", "Biohazard")

            // Update UI with result
            resultTextView.text = classes[maxPos]

        } catch (e: Exception) {
            Toast.makeText(this, "Error running model inference: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadModelFile(modelFilename: String): ByteBuffer {
        val assetFileDescriptor = assets.openFd(modelFilename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close() // Close the model interpreter
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun ImageProxy.toBitmap(): Bitmap {
        val buffer = this.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
}