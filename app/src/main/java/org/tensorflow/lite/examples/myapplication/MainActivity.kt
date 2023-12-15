package org.tensorflow.lite.examples.myapplication
import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.examples.myapplication.ml.ModeloConvertido
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val model = ModeloConvertido.newInstance(this)

        val byteBuffer = loadByteBufferFromAsset(this, "data_buffer2.bin")
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 4, 4), DataType.FLOAT32)

            inputFeature0.loadBuffer(byteBuffer)

            Log.d("MainActivity", "Entrada: ${inputFeature0.floatArray.contentToString()}")

            val outputs = model.process(inputFeature0)
            val outputData = outputs.outputFeature0AsTensorBuffer.floatArray
            Log.d("MainActivity", "Resultaeeeedo da inferência: ${outputData.contentToString()}")



        model.close()
        Log.d("MainActivity", "Modelo fechado.")
    }

    fun loadByteBufferFromAsset(context: Context, assetName: String): ByteBuffer {
        context.assets.open(assetName).use { inputStream ->
            val bytes = inputStream.readBytes()
            return ByteBuffer.wrap(bytes)
        }
    }
}
